**Title:**
Cryptocurrency Price Variation Predictor

**README.md:**

# Cryptocurrency Price Variation Predictor

This repository contains a Python script for predicting the price variation of a cryptocurrency (ACHUSDT) for the next 24 hours using historical data and machine learning techniques.

## Overview

The script utilizes Binance API to fetch cryptocurrency data, preprocesses the data, trains Gradient Boosting Regressor models, and predicts the price variation for the next 24 hours.

## Prerequisites

Make sure you have the following libraries installed:

- requests
- pandas
- scikit-learn
- optuna

You can install these libraries using pip:

```
pip install requests pandas scikit-learn optuna
```

## Usage

1. Clone this repository:

```
git clone https://github.com/your-username/cryptocurrency-price-predictor.git
```

2. Navigate to the cloned repository:

```
cd cryptocurrency-price-predictor
```

3. Run the script:

```
python predict_price_variation.py
```

## Description

The script performs the following tasks:

1. Fetches historical cryptocurrency data from Binance API.
2. Prepares the data for training by calculating actual minimum and maximum variations, and incorporating Bitcoin daily variations.
3. Trains two Gradient Boosting Regressor models for minimum and maximum price variations.
4. Saves the trained models using pickle.
5. Predicts the price variations for the next 24 hours based on the latest data.

## Author

[Your Name](https://github.com/your-username)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
