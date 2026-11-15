# FunnyFlight Tracker ✈️

An experimental Python project for collecting aircraft observations around Prague and exploring **unsupervised anomaly detection** to identify potentially uncommon aircraft.

The project combines live ADS-B data collection, feature engineering, an Isolation Forest model, and an experimental aircraft rarity scoring system.

**Status: Work in Progress.** Data collection and initial ML training have been implemented. Model validation and automatic Telegram notifications for rare aircraft are not yet complete.

## Project Overview

The goal is to identify aircraft that appear unusual compared with previously collected observations near Prague.

The project consists of several components:

- **Data collection:** Retrieves nearby aircraft observations using the ADS-B API.
- **Data processing:** Cleans observations, normalizes selected aircraft type codes, and stores historical data.
- **Feature engineering:** Calculates frequency-based features for aircraft types, operator codes, and their combinations.
- **Anomaly detection:** Uses scikit-learn's Isolation Forest to score potentially uncommon observations.
- **Optional data enrichment:** Retrieves additional flight information from the Flightradar24 API.
- **Telegram bot prototype:** Displays nearby aircraft on request. Automatic anomaly alerts have not yet been implemented.

## Machine Learning Approach

The initial model uses three numerical features:

1. Aircraft type rarity, derived from its frequency in the collected dataset.
2. Operator code rarity, derived from its observed frequency.
3. Aircraft type–operator pair rarity, derived from the frequency of that combination.

An **Isolation Forest** is trained on these features to assign anomaly scores.

The application also computes a scaled `rarity_score` to rank observations. This score is experimental: **it is not a probability or a validated measure of real-world aircraft rarity**.

The current model does not use flight routes as input features. The available historical dataset is limited, and model performance has not yet been formally validated.

## Technologies

- Python 3.12
- pandas and NumPy
- scikit-learn (Isolation Forest)
- requests
- ADS-B API
- Flightradar24 API (optional)
- python-telegram-bot
- joblib

## Installation

Clone the repository and open the project directory:

```bash
git clone https://github.com/bhoriunov/FunnyFlight-Tracker.git
cd FunnyFlight-Tracker
```

Create a Python 3.12 virtual environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Data Collection and Inference

Run the pipeline once:

```bash
python pipeline.py
```

Or collect observations repeatedly, with a 60-second interval between runs:

```bash
python -u pipeline.py --interval 60
```

Stop continuous collection using `Ctrl+C`.

The pipeline retrieves aircraft observations near central Prague, computes rarity scores using the saved model, and writes the latest results to `live_snapshot.csv`.

Historical observations are accumulated in `training_data.csv`. Repeated observations of the same aircraft and callsign within one UTC day are deduplicated.

The pipeline uses `model_artifacts.joblib`, which contains a previously trained model and associated frequency statistics.

The core ADS-B collection does not require an API key. Optional Flightradar24 enrichment requires a valid API key; without one, enrichment requests may fail while the main collection and scoring workflow continues.

## Model Training

After collecting training observations, retrain the model with:

```bash
python train.py
```

The training script reads `training_data.csv`, constructs frequency-based features, fits an Isolation Forest model, and saves:

- `model_artifacts.joblib` — trained model and supporting statistics
- `result.csv` — training observations ranked by anomaly score

**Important:** `training_data.csv` is generated locally and is not included in this repository. Retraining requires collecting the dataset first.

Running `train.py` overwrites existing model artifacts and `result.csv`.

## Optional Telegram Bot

The Telegram bot is a separate prototype that can display aircraft near Prague using the `/planes` command.

To use it, copy `.env.example` to `.env` and provide your own Telegram bot token:

```dotenv
TELEGRAM_BOT_TOKEN=your_token_here
FR24_API_KEY=
```

Run:

```bash
python bot.py
```

The bot is **not yet connected to the ML rarity scoring pipeline** and does not send automatic rare-aircraft notifications.

For optional Flightradar24 enrichment, configure `FR24_API_KEY` in your local `.env` file.

Never commit real API keys or tokens.

## Testing

Run the offline regression tests:

```bash
python -B -m unittest discover -s tests -v
```

The tests use mocked API responses and temporary files, without requiring live external API calls or AWS access.

## Current Limitations and Future Work

- The model was trained on a limited period of aircraft observations.
- Rarity scores have not been validated against independently labeled examples.
- Operator codes are inferred from callsigns and may not always identify the actual operator accurately.
- The current model does not incorporate flight routes, time of day, or seasonality.
- Automatic Telegram notifications for rare aircraft remain unfinished.

Potential future improvements include collecting a longer historical dataset, comparing Isolation Forest with simpler frequency-based baselines, validating detections, and integrating the model with the Telegram bot.

## Project Status

This is a personal learning and experimentation project, not a production flight-tracking or aviation safety system.