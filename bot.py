import os
import asyncio
import requests

from dotenv import load_dotenv
from telegram import Update, ReplyKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    if message is None:
        return

    keyboard = ReplyKeyboardMarkup(
        [["✈️ Planes"]],
        resize_keyboard=True,
        one_time_keyboard=False,
        is_persistent=True,
    )
    await message.reply_text(
        "Welcome to FunnyFlight Tracker! Tap Planes to find aircraft near Prague.",
        reply_markup=keyboard,
    )


def fetch_route(callsign: str) -> str:
    """Look up a reference route; missing data must not break /planes."""
    try:
        response = requests.get(
            f"https://api.adsbdb.com/v0/callsign/{callsign}", timeout=10
        )
        if response.status_code == 404:
            return "Unknown"
        response.raise_for_status()
        route = response.json()["response"]["flightroute"]
        origin = route["origin"].get("iata_code")
        destination = route["destination"].get("iata_code")
        if not origin or not destination:
            return "Unknown"
        midpoint = (route.get("midpoint") or {}).get("iata_code")
        airports = [origin, midpoint, destination] if midpoint else [origin, destination]
        return "-".join(airports)
    except (requests.exceptions.RequestException, ValueError, KeyError, TypeError, AttributeError) as error:
        print(f"Route lookup failed for {callsign}: {type(error).__name__}", flush=True)
        return "Temporarily unavailable"


async def get_routes(aircraft: list) -> dict:
    # Deduplicate callsigns and limit concurrent requests to the route service.
    callsigns = set()
    for plane in aircraft:
        callsign = (plane.get("flight") or "").strip().upper()
        if (callsign and callsign.isascii() and callsign.isalnum()
                and plane.get("t") not in {"TWR", "GND"}):
            callsigns.add(callsign)

    semaphore = asyncio.Semaphore(3)

    async def lookup(callsign):
        async with semaphore:
            route = await asyncio.to_thread(fetch_route, callsign)
            return callsign, route

    return dict(await asyncio.gather(*(lookup(c) for c in sorted(callsigns))))


async def planes(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    if message is None:
        return

    url = "https://api.adsb.lol/v2/lat/50.0755/lon/14.4378/dist/10"
    await message.reply_text("Searching for aircraft near Prague…")

    try:
        response = await asyncio.to_thread(requests.get, url, timeout=15)
        response.raise_for_status()
        aircraft = response.json()["ac"]

    except requests.exceptions.RequestException:
        await message.reply_text("Unable to fetch aircraft data. Please try again later.")
        return

    if not aircraft:
        await message.reply_text("No aircraft found in this area. Please try again later.")
        return

    routes = await get_routes(aircraft)

    lines = [f"✈️ Planes near Prague: {len(aircraft)}", ""]

    for plane in aircraft:
        callsign = (plane.get("flight") or "").strip() or "Unknown"
        aircraft_type = plane.get("t") or "Unknown"
        aircraft_reg = plane.get("r") or "Unknown"
        route_text = routes.get(callsign.upper(), "Unknown")

        lines.append(
                f"{callsign} - {aircraft_type} - {aircraft_reg}\n"
                f"Route (reference): {route_text}\n"
            )

    text = '\n'.join(lines)

    for pos in range (0, len(text), 4000):
        await message.reply_text(text[pos:pos + 4000])

def main():
    load_dotenv()
    token = os.getenv("TELEGRAM_BOT_TOKEN")

    if not token:
        raise SystemExit("Missing TELEGRAM_BOT_TOKEN in .env")

    app = Application.builder().token(token).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("planes", planes))
    app.add_handler(
        MessageHandler(filters.TEXT & filters.Regex(r"^Start$"), start)
    )

    app.add_handler(
        MessageHandler(filters.TEXT & filters.Regex(r"^✈️ Planes$"), planes)
    )

    app.run_polling()

if __name__ == "__main__":
    main()
