import argparse
from pathlib import Path
import time
import pandas as pd
import requests

from adsb_data import AdsbFetchError, fetch_adsb_data
from predict import score_plane
from dataset_store import append_training_data
from fr24_data import get_summaries_by_callsigns
from fr24_cache import (
    filter_uncached,
    mark_as_cached,
    get_cached_enrichment,
)

RARITY = 40
SNAPSHOT_PATH = Path(__file__).resolve().parent / "live_snapshot.csv"

aircraft_map = {
    # Airbus A320 family
    "A318": "A320",
    "A319": "A320",
    "A320": "A320",
    "A20N": "A320",

    # Airbus A321 family
    "A321": "A321",
    "A21N": "A321",

    # Airbus A330 family
    "A332": "A330",
    "A333": "A330",
    "A338": "A330",
    "A339": "A330",

    # Airbus A350 family
    "A359": "A350",
    "A35K": "A350",

    # Airbus A380
    "A388": "A380",

    # Boeing 737 family
    "B736": "B737",
    "B737": "B737",
    "B738": "B737",
    "B739": "B737",
    "B37M": "B737",
    "B38M": "B737",
    "B39M": "B737",
    "B3XM": "B737",

    # Boeing 747 family
    "B741": "B747",
    "B742": "B747",
    "B743": "B747",
    "B744": "B747",
    "B748": "B747",

    # Boeing 777 family
    "B772": "B777",
    "B77L": "B777",
    "B773": "B777",
    "B77W": "B777",
    "B778": "B777",
    "B779": "B777",

    # Boeing 787 family
    "B788": "B787",
    "B789": "B787",
    "B78X": "B787",

    # Embraer E-Jet family
    "E170": "E190",
    "E75L": "E190",
    "E75S": "E190",
    "E190": "E190",
    "E195": "E190",
    "E290": "E190",
    "E295": "E190",

    # Antonov An-124
    "A124": "AN124",

    # Ilyushin Il-76
    "IL76": "IL76",

    # McDonnell Douglas MD-11
    "MD11": "MD11",
}


def build_live_dataset() -> pd.DataFrame:
    live_df = fetch_adsb_data()

    live_df["hex"] = live_df["hex"].str.upper()

    live_df["model_aircraft"] = live_df["aircraft"].replace(
        aircraft_map
    )

    live_df = live_df.dropna(
        subset=["hex", "aircraft", "operator_code"]
    ).copy()

    return live_df


def score_live_dataset(live_df: pd.DataFrame) -> pd.DataFrame:
    if live_df.empty:
        return live_df.assign(
            anomaly_score=pd.Series(dtype=float),
            rarity_score=pd.Series(dtype=float),
        )
    scores = []

    for _, row in live_df.iterrows():
        result = score_plane(
            row["model_aircraft"],
            row["operator_code"]
        )

        scores.append(result)

    scores_df = pd.DataFrame(scores)

    live_df = live_df.reset_index(drop=True)

    live_df[[
        "anomaly_score",
        "rarity_score"
    ]] = scores_df[[
        "anomaly_score",
        "rarity_score"
    ]]

    ranked_df = live_df.sort_values(
        by="rarity_score",
        ascending=False
    )

    return ranked_df


def run_once() -> bool:
    try:
        live_df = build_live_dataset()
    except AdsbFetchError as error:
        print(f"Skipping snapshot: {error}", flush=True)
        return False
    ranked_df = score_live_dataset(live_df)
    
    top_rare = ranked_df[
        ranked_df["rarity_score"] >= RARITY
    ].head(3)

    uncached_rare = filter_uncached(top_rare)

    summaries = []

    if not uncached_rare.empty:
        rare_callsigns = (
            uncached_rare["callsign"]
            .str.strip()
            .tolist()
        )

        try:
            summaries = get_summaries_by_callsigns(
                rare_callsigns
            )
        except requests.RequestException as error:
            print(
                f"FR24 enrichment failed: {error}",
                flush=True
            )
        else:
            mark_as_cached(uncached_rare, summaries)


    cached_enrichment = get_cached_enrichment(top_rare)

    enriched_df = ranked_df.copy()

    enriched_df["hex"] = (
        enriched_df["hex"]
        .str.strip()
        .str.upper()
    )

    enriched_df["callsign"] = (
        enriched_df["callsign"]
        .str.strip()
    )

    if not cached_enrichment.empty:
        enriched_df = enriched_df.merge(
            cached_enrichment,
            on=["hex", "callsign"],
            how="left"
        )
    else:
        enriched_df["orig_icao"] = pd.NA
        enriched_df["dest_icao"] = pd.NA
        enriched_df["operating_as"] = pd.NA
        enriched_df["fr24_id"] = pd.NA

    training_rows = ranked_df[[
        "hex",
        "callsign",
        "registration",
        "aircraft",
        "model_aircraft",
        "operator_code",
    ]]
    append_training_data(training_rows)

    output_columns = [
        "hex",
        "callsign",
        "registration",
        "aircraft",
        "model_aircraft",
        "operator_code",
        "alt_baro",
        "gs",
        "dst",
        "rarity_score",
        "orig_icao",
        "dest_icao",
        "operating_as",
        "fr24_id",
    ]

    print(enriched_df[output_columns])

    enriched_df[output_columns].to_csv(
        SNAPSHOT_PATH,
        index=False
    )

    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect and score aircraft near Prague")
    parser.add_argument("--interval", type=int, help="Repeat every N seconds (default: one run)")
    args = parser.parse_args()
    if args.interval is not None and args.interval <= 0:
        parser.error("--interval must be positive")
    try:
        if args.interval is None:
            raise SystemExit(0 if run_once() else 1)
        while True:
            run_once()
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nCollection stopped.", flush=True)


if __name__ == "__main__":
    main()
