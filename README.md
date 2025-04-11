# Crypto Price Predictor

A Python script to predict cryptocurrency prices using pre-trained LSTM models.

## Features
- Predicts prices for any cryptocurrency traded on Binance (e.g., BTCUSDT, ETHUSDT).
- Supports two models:
  - Default model: Trained on 21 cryptocurrency pairs, 3 LSTM layers with 100 units.
  - Powerful model: Trained on 36 cryptocurrency pairs, 4 LSTM layers with 150 units.
- Allows multi-day predictions (e.g., 3 days ahead).
- Fetches historical data directly from Binance API.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/nilsonleow/crypto-price-predictor.git
   cd crypto-price-predictor
2. Install dependencies:
   ```bash
   pip install -r requirements.txt

## Usage
Run the script and specify the model using the --powerful flag (optional):
   ```bash
    python predict.py [--powerful]
--powerful: Use the powerful model (36 pairs, 4 LSTM layers with 150 units). If not specified, the default model (21 pairs, 3 LSTM layers with 100 units) is used.

You will be prompted to enter the ticker and prediction period:
Example: BTC 3d (predict BTC price 3 days ahead).

Example: ETH (predict ETH price 1 day ahead, default period).

