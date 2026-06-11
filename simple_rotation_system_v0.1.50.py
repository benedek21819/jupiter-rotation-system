#!/usr/bin/env python3
"""
Simple Asset Rotation System v0.1.50 (Main Version)
================================================
Best performing stable long-only version.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta, timezone
import ccxt

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False

ASSETS = ['BTC', 'ETH', 'SOL', 'HYPE', 'SUI', 'BNB', 'TSLA', 'NVDA', 'SPY', 'QQQ', 'GLD']
MAX_ASSETS = 5
MIN_ALLOCATION = 0.10
INITIAL_CAPITAL = 10_000.0
DATA_SOURCE = "live"
CSV_PATH = "/home/workdir/your_prices.csv"
EXCHANGE = "bybit"
DAYS_OF_HISTORY = 1500

PMOTION_EMA_LENGTH = 21
PMOTION_SD_LENGTH = 30
PMOTION_SD_MULT = 1.5
PMOTION_DEMA_LENGTH = 7
PMOTION_MEDIAN_LENGTH = 2
KAMA_ER_THRESHOLD = 0.3236679226575263
V2_IMPULSIVE_BULLISH = 0.6341537373741729
V2_IMPULSIVE_BEARISH = -1.4055678761345258
HURST_MEAN_REVERSION_THRESHOLD = 0.46317135333742043
HURST_PENALTY = 0.6
TREND_DECAY_THRESHOLD = 49
TREND_DECAY_ROC = -10
TREND_DECAY_FACTOR = 0.5272833353564802
DYN_STOP_LOOKBACK = 10
DYN_STOP_MIN_PMOTION = 27
DYN_STOP_DROP_THRESHOLD = 25
DYN_STOP_PENALTY = 0.19954276551986405
CORRELATION_LOOKBACK = 60
MAX_CORRELATION = 0.82

OUTPUT_DIR = Path("/home/workdir/artifacts")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def calculate_p_motion_trend(series):
    if len(series) < 30: return pd.Series(0.0, index=series.index)
    ema_fast = series.ewm(span=7, adjust=False).mean()
    ema_slow = ema_fast.ewm(span=7, adjust=False).mean()
    dema = 2 * ema_fast - ema_slow
    smoothed = dema.rolling(window=2, min_periods=1).median()
    basis = smoothed.ewm(span=21, adjust=False).mean()
    rolling_std = basis.rolling(window=30, min_periods=15).std()
    upper = basis + rolling_std * 1.5
    lower = basis - rolling_std * 1.5
    score = pd.Series(0.0, index=series.index)
    score[series > upper] = 100.0
    score[series < lower] = -100.0
    neutral = ~((series > upper) | (series < lower))
    if neutral.any():
        distance = (series - basis) / (rolling_std + 1e-8)
        score[neutral] = np.clip(distance * 55, -80, 80)[neutral]
    basis_slope = (basis - basis.shift(5)) / 5
    slope_std = basis_slope.rolling(window=20).std()
    slope_z = basis_slope / (slope_std + 1e-8)
    score = (score + np.clip(slope_z * 35, -25, 25)).clip(-100, 100)
    return score


def calculate_kaufman_efficiency_ratio(close, length=14):
    if len(close) < length + 1: return pd.Series(0.0, index=close.index)
    change = (close - close.shift(length)).abs()
    volatility = (close - close.shift(1)).abs().rolling(window=length).sum()
    return (change / (volatility + 1e-8)).clip(0, 1)


def calculate_v2_impulsive_momentum(close):
    if len(close) < 30: return pd.Series(0.0, index=close.index)
    ema = close.ewm(span=21, adjust=False).mean()
    atr_val = (close - close.shift(1)).abs().rolling(14).mean()
    mom_vel = (close - ema) / (atr_val + 1e-8)
    score = mom_vel * 40
    final = pd.Series(0.0, index=close.index)
    final[score >= V2_IMPULSIVE_BULLISH] = np.clip(score[score >= V2_IMPULSIVE_BULLISH] * 1.8, 0, 100)
    final[score <= V2_IMPULSIVE_BEARISH] = np.clip(score[score <= V2_IMPULSIVE_BEARISH] * 1.8, -100, 0)
    return final.ewm(span=5, adjust=False).mean().clip(-100, 100)


def calculate_hurst_exponent(series, window=120):
    if len(series) < window + 10: return pd.Series(np.nan, index=series.index)
    hurst = pd.Series(np.nan, index=series.index)
    for i in range(window, len(series)):
        w = series.iloc[i-window:i].values
        mean_adj = w - np.mean(w)
        cum = np.cumsum(mean_adj)
        R = np.max(cum) - np.min(cum)
        S = np.std(w)
        hurst.iloc[i] = np.log(R / S) / np.log(window) if S > 0 else 0.5
    return hurst


def detect_market_regime(avg_pmotion, avg_hurst, vol_regime, market_breadth, gold_vs_equity):
    if avg_pmotion > 18 and market_breadth > 0.4 and vol_regime <= 1:
        return 0, "Bullish"
    if vol_regime == 2 and (avg_pmotion < 5 or market_breadth < 0.25 or gold_vs_equity > 0.05 or avg_hurst < 0.40):
        return 3, "Crisis / Defensive"
    if vol_regime == 2: return 2, "High Volatility"
    return 1, "Neutral"


def fetch_live_prices(assets=None, days=800, exchange='bybit'):
    if assets is None: assets = ASSETS
    print(f"\n[Data] Fetching live data...")
    crypto_assets = [a for a in assets if a in ['BTC','ETH','SOL','HYPE','SUI','BNB','TAO']]
    stock_assets = [a for a in assets if a not in ['BTC','ETH','SOL','HYPE','SUI','BNB','TAO']]
    all_data = {}

    if 'BTC' in crypto_assets:
        print("  Fetching BTC...")
        fetched = False
        try:
            ex = getattr(ccxt, exchange)()
            ex.load_markets()
            since = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)
            for quote in ['USDT', 'USDC']:
                try:
                    ohlcv = ex.fetch_ohlcv(f"BTC/{quote}", '1d', since=since, limit=1000)
                    if ohlcv and len(ohlcv) >= 40:
                        df = pd.DataFrame(ohlcv, columns=['timestamp','open','high','low','close','volume'])
                        df['Date'] = pd.to_datetime(df['timestamp'], unit='ms').dt.date
                        df = df.set_index('Date')
                        all_data['BTC'] = df[['high','low','close']]
                        print("    BTC: \u2713 ccxt")
                        fetched = True
                        break
                except: pass
        except: pass

        if not fetched and HAS_YFINANCE:
            try:
                btc = yf.download("BTC-USD", period=f"{days}d", interval="1d", progress=False)
                if not btc.empty:
                    btc = btc[['High','Low','Close']].dropna()
                    btc.columns = ['high','low','close']
                    btc.index = pd.to_datetime(btc.index).date
                    all_data['BTC'] = btc
                    print("    BTC: \u2713 yfinance")
                    fetched = True
            except: pass
        if not fetched: print("    BTC: \u2717 Failed")
        crypto_assets = [a for a in crypto_assets if a != 'BTC']

    if crypto_assets:
        print(f"  Fetching crypto: {crypto_assets}")
        try:
            ex = getattr(ccxt, exchange)()
            ex.load_markets()
            since = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)
            for asset in crypto_assets:
                for quote in ['USDT','USDC']:
                    try:
                        ohlcv = ex.fetch_ohlcv(f"{asset}/{quote}", '1d', since=since, limit=1000)
                        if ohlcv and len(ohlcv) >= 40:
                            df = pd.DataFrame(ohlcv, columns=['timestamp','open','high','low','close','volume'])
                            df['Date'] = pd.to_datetime(df['timestamp'], unit='ms').dt.date
                            df = df.set_index('Date')
                            all_data[asset] = df[['high','low','close']]
                            break
                    except: pass
        except: pass

    if stock_assets and HAS_YFINANCE:
        print(f"  Fetching stocks: {stock_assets}")
        try:
            tickers = " ".join(stock_assets)
            data = yf.download(tickers, period=f"{days}d", interval="1d", progress=False, group_by='ticker')
            for asset in stock_assets:
                try:
                    df = data[asset] if len(stock_assets) > 1 else data
                    df = df[['High','Low','Close']].dropna()
                    df.columns = ['high','low','close']
                    df.index = pd.to_datetime(df.index).date
                    all_data[asset] = df
                except: pass
        except Exception as e: print(f"  yfinance error: {e}")

    if not all_data: raise RuntimeError("No assets fetched.")

    prices_dict = {}
    for asset, df in all_data.items():
        prices_dict[f"{asset}_high"] = df['high']
        prices_dict[f"{asset}_low"] = df['low']
        prices_dict[f"{asset}_close"] = df['close']

    prices = pd.DataFrame(prices_dict).sort_index().ffill().dropna(how='all')
    prices.index = pd.to_datetime(prices.index)
    prices.index.name = 'Date'
    loaded = [col.replace('_close','') for col in prices.columns if '_close' in col]
    print(f"Loaded: {loaded}")
    return prices, loaded


def load_from_csv(path):
    prices = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
    prices = prices.sort_index().ffill().dropna(how='all')
    loaded = [col.replace('_close','') for col in prices.columns if '_close' in col]
    return prices, loaded


def calculate_btc_performance(prices_df, initial_capital=10000.0):
    if 'BTC_close' not in prices_df.columns:
        return None, None, None
    btc = prices_df['BTC_close'].dropna()
    if len(btc) < 2:
        return None, None, None
    ret = (btc.iloc[-1] / btc.iloc[0] - 1) * 100
    days = (btc.index[-1] - btc.index[0]).days
    cagr = ((btc.iloc[-1] / btc.iloc[0]) ** (1 / (days/365.25)) - 1) * 100
    eq = btc / btc.iloc[0] * initial_capital
    mdd = ((eq - eq.cummax()) / eq.cummax() * 100).min()
    return ret, cagr, mdd


def run_pmotion_system(prices_df, asset_list):
    print(f"\nRunning P-Motion Trend System v0.1.50 on {len(asset_list)} assets...")

    close_prices = pd.DataFrame({asset: prices_df[f"{asset}_close"] for asset in asset_list})

    pmotion_scores = pd.DataFrame(index=close_prices.index, columns=asset_list)
    kama_er_df = pd.DataFrame(index=close_prices.index, columns=asset_list)
    v2_impulsive_df = pd.DataFrame(index=close_prices.index, columns=asset_list)
    hurst_df = pd.DataFrame(index=close_prices.index, columns=asset_list)

    for asset in asset_list:
        pmotion_scores[asset] = calculate_p_motion_trend(close_prices[asset])
        kama_er_df[asset] = calculate_kaufman_efficiency_ratio(close_prices[asset])
        v2_impulsive_df[asset] = calculate_v2_impulsive_momentum(close_prices[asset])
        hurst_df[asset] = calculate_hurst_exponent(close_prices[asset])

    n = len(close_prices)
    allocations = pd.DataFrame(0.0, index=close_prices.index, columns=asset_list)
    equity = pd.Series(10000.0, index=close_prices.index)

    for i in range(30, n):
        date = close_prices.index[i]
        current_pmotion = pmotion_scores.iloc[i]
        current_er = kama_er_df.iloc[i]
        current_v2 = v2_impulsive_df.iloc[i]
        current_hurst = hurst_df.iloc[i]

        avg_pmotion = current_pmotion.mean()
        avg_hurst = current_hurst.mean()
        market_breadth = (current_pmotion > 0).sum() / len(asset_list)

        gold_vs_equity = 0
        if 'GLD_close' in close_prices.columns and 'SPY_close' in close_prices.columns and i > 20:
            gold_vs_equity = (close_prices['GLD_close'].iloc[i] / close_prices['GLD_close'].iloc[i-20] - 1) - \
                             (close_prices['SPY_close'].iloc[i] / close_prices['SPY_close'].iloc[i-20] - 1)

        regime_id, regime_name = detect_market_regime(avg_pmotion, avg_hurst, 1, market_breadth, gold_vs_equity)

        strong_trend_mask = current_er >= KAMA_ER_THRESHOLD
        gated_scores = current_pmotion.copy()
        gated_scores[~strong_trend_mask] = -999

        v2_bullish = (current_v2 >= V2_IMPULSIVE_BULLISH).reindex(gated_scores.index).fillna(False)
        v2_bearish = (current_v2 <= V2_IMPULSIVE_BEARISH).reindex(gated_scores.index).fillna(False)
        gated_scores[~ (v2_bullish | v2_bearish)] = -999

        scaling_factor = np.clip(current_er * 2 + 0.5, a_min=None, a_max=2.5)
        final_scores = gated_scores * scaling_factor

        for asset in asset_list:
            h_val = current_hurst[asset]
            if not pd.isna(h_val) and h_val < HURST_MEAN_REVERSION_THRESHOLD:
                final_scores[asset] *= HURST_PENALTY

        pmotion_roc = (current_pmotion - current_pmotion.shift(5)) / 5
        decay_factor = np.where(
            (current_pmotion > TREND_DECAY_THRESHOLD) & (pmotion_roc < TREND_DECAY_ROC),
            TREND_DECAY_FACTOR, 1.0
        )
        final_scores = final_scores * decay_factor

        pmotion_peak = current_pmotion.rolling(window=10).max()
        pmotion_drop = pmotion_peak - current_pmotion
        stop_loss_factor = np.where(
            (current_pmotion > 27) & (pmotion_drop > 25),
            DYN_STOP_PENALTY, 1.0
        )
        final_scores = final_scores * stop_loss_factor

        ranked = final_scores.sort_values(ascending=False)
        positive = ranked[ranked > 0]
        selected = positive.head(MAX_ASSETS).index.tolist()

        new_weights = pd.Series(0.0, index=asset_list)
        if selected:
            selected_scores = final_scores[selected].clip(lower=0)
            if selected_scores.sum() > 0:
                new_weights[selected] = selected_scores / selected_scores.sum()

        allocations.loc[date] = new_weights

        if i < n - 1:
            next_ret = (close_prices.iloc[i + 1] / close_prices.iloc[i] - 1.0)
            port_ret = (new_weights * next_ret).sum()
            equity.iloc[i + 1] = equity.iloc[i] * (1 + port_ret)

    print("Backtest completed.")
    return allocations, equity, pmotion_scores


if __name__ == "__main__":
    print("=" * 75)
    print("SIMPLE ASSET ROTATION SYSTEM v0.1.50 (Main Version)")
    print("=" * 75)

    if DATA_SOURCE == "live":
        prices_df, active_assets = fetch_live_prices(assets=ASSETS, days=DAYS_OF_HISTORY, exchange=EXCHANGE)
    else:
        prices_df, active_assets = load_from_csv(CSV_PATH)

    if len(active_assets) == 0:
        print("No assets loaded.")
        exit()

    print(f"\nActive assets: {active_assets}")

    allocations, equity, pmotion_scores = run_pmotion_system(prices_df, active_assets)

    final_equity = equity.iloc[-1]
    total_return = (final_equity / 10000 - 1) * 100
    days = (equity.index[-1] - equity.index[0]).days
    years = days / 365.25
    cagr = ((final_equity / 10000) ** (1 / years) - 1) * 100
    max_dd = ((equity - equity.cummax()) / equity.cummax() * 100).min()

    print("\n" + "=" * 70)
    print("STRATEGY PERFORMANCE v0.1.50")
    print("=" * 70)
    print(f"Final Equity : ${final_equity:,.2f}")
    print(f"Total Return : {total_return:+.1f}%")
    print(f"CAGR         : {cagr:+.1f}%")
    print(f"Max Drawdown : {max_dd:.1f}%")

    # BTC Benchmark
    btc_ret, btc_cagr, btc_mdd = calculate_btc_performance(prices_df)
    if btc_ret is not None:
        print("\n" + "=" * 70)
        print("BTC BENCHMARK")
        print("=" * 70)
        print(f"Total Return : {btc_ret:+.1f}%")
        print(f"CAGR         : {btc_cagr:+.1f}%")
        print(f"Max Drawdown : {btc_mdd:.1f}%")

    # Recent Allocations (last 10 days)
    print("\nRecent Allocations (last 10 days):")
    print(allocations.tail(10).round(3))

    # Current Allocation (clean %)
    print("\n" + "=" * 70)
    print("CURRENT ALLOCATION (as of last date)")
    print("=" * 70)

    latest = allocations.iloc[-1]
    latest = latest[latest > 0.005].sort_values(ascending=False)

    if len(latest) > 0:
        for asset, weight in latest.items():
            print(f"  {asset:6s} : {weight * 100:6.2f} %")
        print(f"\n  Total Allocated : {latest.sum() * 100:.2f} %")
    else:
        print("  No positions (system in cash)")

    print("\n" + "=" * 70)

    allocations.to_csv(OUTPUT_DIR / "v0.1.50_allocations.csv")
    equity.to_frame('Equity').to_csv(OUTPUT_DIR / "v0.1.50_equity.csv")
    print(f"Files saved to {OUTPUT_DIR}")
    print("\nDone.")