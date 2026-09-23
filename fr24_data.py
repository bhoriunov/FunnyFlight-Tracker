import os
from dotenv import load_dotenv
import requests
import pandas as pd

def get_headers() -> dict:
    load_dotenv()

    token = os.getenv("FR24_API_KEY")

    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
        "Accept-Version": "v1",
    }

def fetch_fr24_live() -> list:
    url = "https://fr24api.flightradar24.com/api/live/flight-positions/light"

    params = {
        "bounds": "50.30,49.90,14.10,14.80",
        "limit": 3,
    }

    response = requests.get(
        url,
        headers=get_headers(),
        params=params,
        timeout=15
    )

    response.raise_for_status()

    data = response.json()

    return data["data"]

def get_flight_summary(flight_id: str) -> dict | None:
    summary_url = "https://fr24api.flightradar24.com/api/flight-summary/light"

    summary_params = {
        "flight_ids": flight_id
    }

    summary_response = requests.get(
        summary_url,
        headers=get_headers(),
        params=summary_params,
        timeout=15
    )

    summary_response.raise_for_status()
    summary_data = summary_response.json()

    if not summary_data["data"]:
        return None

    return summary_data["data"][0]

if __name__ == "__main__":
    flights = fetch_fr24_live()

    summaries = []

    for flight in flights:
        summary = get_flight_summary(flight["fr24_id"])

        if summary is not None:
            summaries.append(summary)

    summary_df = pd.DataFrame(summaries)

    print(summary_df[[
        "callsign",
        "operating_as",
        "type",
        "reg",
        "orig_icao",
        "dest_icao",
        "hex",
    ]])
