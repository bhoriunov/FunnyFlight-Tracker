from pathlib import Path
import pandas as pd


TRAINING_PATH = Path(__file__).resolve().parent / "training_data.csv"

def append_training_data(new_df: pd.DataFrame) -> None:
    if new_df.empty:
        return
    new_df = new_df.copy()
    now = pd.Timestamp.now(tz="UTC")

    new_df["seen_at"] = now.isoformat()
    new_df["flight_date"] = now.strftime("%Y-%m-%d")

    new_df["callsign"] = new_df["callsign"].str.strip()
    new_df["hex"] = new_df["hex"].str.strip().str.upper()

    if TRAINING_PATH.exists():
        old_df = pd.read_csv(TRAINING_PATH, dtype={"hex": "string", "callsign": "string", "flight_date": "string"})

        old_df["flight_date"] = old_df["flight_date"].astype(str)
        old_df["callsign"] = old_df["callsign"].str.strip()
        old_df["hex"] = old_df["hex"].str.strip().str.upper()
    else:
        old_df = pd.DataFrame()

    combined_df = pd.concat(
        [old_df, new_df],
        ignore_index=True
    )

    combined_df = combined_df.drop_duplicates(
        subset=["flight_date", "hex", "callsign"],
        keep="last"
    )

    combined_df.to_csv(
        TRAINING_PATH,
        index=False
    )
