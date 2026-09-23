from pathlib import Path

import requests
import pandas as pd

operator_map = {
    "TVS": "Smartwings",
    "EZY": "easyJet",
    "KAL": "Korean Air",
    "NJE": "NetJets Europe",
}

url = "https://api.adsb.lol/v2/lat/50.0755/lon/14.4378/dist/10"

INPUT_COLUMNS = ["hex", "flight", "r", "t", "lat", "lon", "alt_baro", "gs", "dst"]


class AdsbFetchError(RuntimeError):
    """The API could not provide a usable snapshot."""


def fetch_adsb_data() -> pd.DataFrame:
    try:
        response = requests.get(url=url, timeout=15)
        response.raise_for_status()
        data = response.json()
    except (requests.exceptions.RequestException, ValueError) as error:
        raise AdsbFetchError(f"ADS-B request failed: {error}") from error

    if not isinstance(data, dict) or not isinstance(data.get("ac"), list):
        raise AdsbFetchError("ADS-B response must contain an aircraft list")
    if any(not isinstance(plane, dict) for plane in data["ac"]):
        raise AdsbFetchError("ADS-B response contains an invalid aircraft record")

    # Keep a stable schema even when the list or optional fields are empty.
    df = pd.DataFrame(data["ac"]).reindex(columns=INPUT_COLUMNS)
    for column in ("hex", "flight", "r", "t"):
        df[column] = df[column].astype("string").str.strip().replace("", pd.NA)
    df["flight"] = df["flight"].str.upper()
    df["hex"] = df["hex"].str.upper()
    df["t"] = df["t"].str.upper()
    clean_df = df[df["t"].notna() & ~df["t"].isin(["GND", "TWR"])].copy()
    clean_df["alt_baro"] = pd.to_numeric(
        clean_df["alt_baro"].replace("ground", 0), errors="coerce"
    )
    clean_df = clean_df.rename(columns={
        "t": "aircraft", "r": "registration", "flight": "callsign",
    })
    codes = clean_df["callsign"].str[:3]
    clean_df["operator_code"] = codes.where(codes.str.fullmatch(r"[A-Z]{3}", na=False))
    clean_df["operator"] = clean_df["operator_code"].map(operator_map)
    return clean_df


if __name__ == "__main__":
    try:
        clean_df = fetch_adsb_data()
    except AdsbFetchError as error:
        raise SystemExit(str(error)) from error

    print(clean_df[[
        "callsign",
        "operator_code",
        "operator",
        "aircraft"
    ]])

    clean_df[
        ["hex", "registration", "aircraft", "operator_code"]
    ].to_csv(Path(__file__).resolve().parent / "airline_cache.csv", index=False)
