from pathlib import Path

import pandas as pd


CACHE_PATH = Path(__file__).resolve().parent / "fr24_cache.csv"

CACHE_COLUMNS = [
    "flight_date",
    "hex",
    "callsign",
    "looked_up",
    "orig_icao",
    "dest_icao",
    "operating_as",
    "fr24_id",
]


def _today() -> str:
    return pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d")


def _load_cache() -> pd.DataFrame:
    if not CACHE_PATH.exists():
        return pd.DataFrame(columns=CACHE_COLUMNS)

    cache_df = pd.read_csv(CACHE_PATH)

    # Підтримка старого cache-файла
    for column in CACHE_COLUMNS:
        if column not in cache_df.columns:
            if column == "looked_up":
                cache_df[column] = False
            else:
                cache_df[column] = pd.NA

    cache_df["flight_date"] = cache_df["flight_date"].astype(str)
    cache_df["hex"] = cache_df["hex"].astype(str).str.strip().str.upper()
    cache_df["callsign"] = cache_df["callsign"].astype(str).str.strip()

    cache_df["looked_up"] = (
        cache_df["looked_up"]
        .astype(str)
        .str.lower()
        .isin(["true", "1", "yes"])
    )

    return cache_df[CACHE_COLUMNS]


def filter_uncached(top_rare: pd.DataFrame) -> pd.DataFrame:
    if top_rare.empty:
        return top_rare

    rows = top_rare.copy()

    rows["flight_date"] = _today()
    rows["hex"] = rows["hex"].str.strip().str.upper()
    rows["callsign"] = rows["callsign"].str.strip()

    cache_df = _load_cache()

    completed_cache = cache_df.loc[
        cache_df["looked_up"].eq(True)
    ]

    cached_keys = set(
        zip(
            completed_cache["flight_date"],
            completed_cache["hex"],
            completed_cache["callsign"],
        )
    )

    is_cached = rows.apply(
        lambda row: (
            row["flight_date"],
            row["hex"],
            row["callsign"],
        ) in cached_keys,
        axis=1,
    )

    return rows.loc[~is_cached].drop(
        columns=["flight_date"]
    ).copy()


def mark_as_cached(
    rows: pd.DataFrame,
    summaries: list[dict],
) -> None:
    if rows.empty:
        return

    requested = rows.copy()

    requested["flight_date"] = _today()
    requested["hex"] = (
        requested["hex"]
        .str.strip()
        .str.upper()
    )
    requested["callsign"] = (
        requested["callsign"]
        .str.strip()
    )

    cache_rows = requested[[
        "flight_date",
        "hex",
        "callsign",
    ]].copy()

    cache_rows["looked_up"] = True

    if summaries:
        summary_df = pd.DataFrame(summaries)

        summary_df["hex"] = (
            summary_df["hex"]
            .str.strip()
            .str.upper()
        )

        summary_df = summary_df.drop_duplicates(
            subset=["hex"],
            keep="last",
        )

        summary_df = summary_df.reindex(columns=[
            "hex",
            "orig_icao",
            "dest_icao",
            "operating_as",
            "fr24_id",
        ])

        cache_rows = cache_rows.merge(
            summary_df,
            on="hex",
            how="left",
        )
    else:
        cache_rows["orig_icao"] = pd.NA
        cache_rows["dest_icao"] = pd.NA
        cache_rows["operating_as"] = pd.NA
        cache_rows["fr24_id"] = pd.NA

    old_cache = _load_cache()

    combined = pd.concat(
        [old_cache, cache_rows],
        ignore_index=True,
    )

    combined = combined.drop_duplicates(
        subset=[
            "flight_date",
            "hex",
            "callsign",
        ],
        keep="last",
    )

    combined[CACHE_COLUMNS].to_csv(
        CACHE_PATH,
        index=False,
    )


def get_cached_enrichment(
    top_rare: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "hex",
        "callsign",
        "orig_icao",
        "dest_icao",
        "operating_as",
        "fr24_id",
    ]

    if top_rare.empty:
        return pd.DataFrame(columns=columns)

    rows = top_rare.copy()

    rows["flight_date"] = _today()
    rows["hex"] = (
        rows["hex"]
        .str.strip()
        .str.upper()
    )
    rows["callsign"] = (
        rows["callsign"]
        .str.strip()
    )

    cache_df = _load_cache()

    cache_df = cache_df[
        cache_df["looked_up"].eq(True)
    ]

    matched = rows[[
        "flight_date",
        "hex",
        "callsign",
    ]].merge(
        cache_df,
        on=[
            "flight_date",
            "hex",
            "callsign",
        ],
        how="left",
    )

    return matched[columns]