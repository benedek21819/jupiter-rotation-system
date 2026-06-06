# Jupiter Rotation System

Deterministic Python implementation of the Jupiter Rotation System for asset allocation on Solana via Jupiter DEX.

## Overview

This repository contains a fully deterministic, rules-based asset rotation system with multiple risk management layers, regime detection, and execution instructions for Jupiter.

## Key Features

- **Regime Confidence Scaling**: Final Exposure = Base × Regime Confidence
- **EMA(14) Regime Smoothing**
- **Drawdown Speed Filter**
- **Leadership Decay Filter**
- **Asymmetric Entry Bias** (+10% to clear #1 leader)
- **Acceleration Phase Detection**
- **Meta Stability Governor**
- **Conviction Dispersion Weighting**
- **Cluster-Level Exposure Caps**
- **Signal Diversity Score**
- **Jupiter Execution Instructions Generator**

## Structure

- `jupiter_rotation_system.py`: Core engine with all rules (deterministic)
- `jupiter_signal_generator.py`: Converts signals into ready-to-execute Jupiter swap instructions

## Usage

See the example in the code or run the signal generator with your daily data.

## Data Input

The system is designed to accept daily feature vectors (indicator scores, macro data, portfolio state). You can feed it manually or via Google Sheets / API in future versions.

## Disclaimer

This is a personal research / educational project. Not financial advice. Trade at your own risk.