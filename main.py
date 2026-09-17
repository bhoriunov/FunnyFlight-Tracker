import time 
import requests

from pprint import pprint

url = "https://api.adsb.lol/v2/lat/50.0755/lon/14.4378/dist/10"

try:
    while True:
        try:
            response = requests.get(url, timeout=15)
            response.raise_for_status()

            data = response.json()
            aircraft = data["ac"]

            print(f"Count of found aircrafts: {len(aircraft)}")

            for plane in aircraft:
                callsign = (plane.get("flight") or "").strip() or "Unknown"
                aircraft_type = plane.get("t") or "Unknown"
                aircraft_reg = plane.get("r") or "Unknown"

                print(f"{callsign} - {aircraft_type} - {aircraft_reg}")
            
        except requests.exceptions.RequestException as error:
            print(f"Request error: {error}")
            
        time.sleep(60)

except KeyboardInterrupt:
    print("\n Monitoring ended ")