# FunnyFlight-Tracker

Requires Python 3.12. Install dependencies in a fresh virtual environment with
`python -m pip install -r requirements.txt`.

## Collector

- `python pipeline.py`: one snapshot; exits 1 on an API failure.
- `python -u pipeline.py --interval 60`: repeat, waiting 60 seconds after each run;
  stop with Ctrl+C. API failures skip the snapshot and retry next cycle.
- Empty valid responses write an empty snapshot with column headers. API failures
  preserve the last snapshot. Empty datasets do not change training history.
- Run only one collector at a time. Do not combine interval mode with cron.
- This command does not install a background service; it must remain running.

The model and CSV paths are relative to the project directory, regardless of the
working directory. `model_artifacts.joblib` is required and is included in git.
Use only trusted model files. Keep dependency versions aligned with training.
The Mac `.venv` is not portable: create a new environment on Linux.

`training_data.csv` is created automatically; `live_snapshot.csv` is overwritten
with the latest successful result. Both are local generated data, excluded from git.
The collector does not need API keys.

## Optional scripts

Copy `.env.example` to `.env` and fill `TELEGRAM_BOT_TOKEN` for `python bot.py`,
or `FR24_API_KEY` for `python fr24_data.py`. Never commit real secrets.
The bot is separate from the model pipeline. `main.py` only prints live aircraft.
`train.py` uses a built-in sample, not `training_data.csv`, and overwrites the
model and `result.csv` when run (including when imported).

## Offline regression checks

`python -B -m unittest discover -s tests -v`

Tests use mocked API responses and temporary output files. No AWS access required.
