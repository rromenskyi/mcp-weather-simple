from __future__ import annotations

import os
import secrets
from datetime import date, datetime, timezone as _tz
from zoneinfo import ZoneInfo

import httpx
from mcp.server.fastmcp import FastMCP

TRANSPORT = os.getenv("MCP_TRANSPORT", "stdio")
HOST = os.getenv("MCP_HOST", "0.0.0.0")
PORT = int(os.getenv("MCP_PORT", "8000"))
AUTH_TOKEN = os.getenv("MCP_AUTH_TOKEN", "").strip()

mcp = FastMCP("weather", host=HOST, port=PORT, stateless_http=True)

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

WEATHER_CODES = {
    0: "clear sky",
    1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "fog", 48: "depositing rime fog",
    51: "light drizzle", 53: "moderate drizzle", 55: "dense drizzle",
    56: "light freezing drizzle", 57: "dense freezing drizzle",
    61: "slight rain", 63: "moderate rain", 65: "heavy rain",
    66: "light freezing rain", 67: "heavy freezing rain",
    71: "slight snow", 73: "moderate snow", 75: "heavy snow",
    77: "snow grains",
    80: "slight rain showers", 81: "moderate rain showers", 82: "violent rain showers",
    85: "slight snow showers", 86: "heavy snow showers",
    95: "thunderstorm", 96: "thunderstorm with slight hail", 99: "thunderstorm with heavy hail",
}


async def _geocode(city: str, country_code: str | None = None) -> dict:
    # Pull a few candidates so we can filter by `country_code` client-side —
    # Open-Meteo's geocoding endpoint does not accept a country filter, it
    # just matches `name` across the whole globe. `count=10` is cheap and
    # covers the vast majority of ambiguous city names (Moscow, Paris,
    # Springfield, etc.).
    params: dict[str, str | int] = {"name": city, "count": 10, "language": "en"}
    async with httpx.AsyncClient(timeout=15.0) as client:
        r = await client.get(GEOCODE_URL, params=params)
        r.raise_for_status()
        data = r.json()
    results = data.get("results") or []
    if country_code:
        cc_upper = country_code.strip().upper()
        filtered = [h for h in results if (h.get("country_code") or "").upper() == cc_upper]
        if filtered:
            results = filtered
    if not results:
        raise ValueError(
            f"City not found: {city}" + (f" in {country_code}" if country_code else "")
        )
    hit = results[0]
    return {
        "name": hit["name"],
        "country": hit.get("country"),
        "country_code": hit.get("country_code"),
        "admin1": hit.get("admin1"),  # state / region, helps disambiguate "Springfield, IL" vs "Springfield, MA"
        "latitude": hit["latitude"],
        "longitude": hit["longitude"],
        "timezone": hit.get("timezone"),
        # List of postal codes tied to this hit. Useful to confirm a
        # zipcode-driven lookup actually resolved the expected area —
        # e.g. "90210" → name "Beverly Hills", postcodes contains 90210.
        "postcodes": hit.get("postcodes"),
    }


def _day_label(target: date, today: date) -> str:
    """Map an ISO date to a relative label the LLM can anchor on."""
    delta = (target - today).days
    if delta == 0:
        return "today"
    if delta == 1:
        return "tomorrow"
    if delta == -1:
        return "yesterday"
    if delta > 1:
        return f"in {delta} days"
    return f"{-delta} days ago"


@mcp.tool()
async def geocode_city(city: str, country_code: str | None = None) -> dict:
    """Resolve a city name or postal code to lat/lon, country and timezone.

    `city` accepts plain names ("Kyiv", "Paris") and numeric postal
    codes ("90210", "10001"). Support varies by country — US/DE/FR
    zipcodes resolve cleanly, UK postcodes (e.g. "SW1A 1AA") are
    **not** indexed by Open-Meteo and return no result. For a postal
    code always pass `country_code` too — e.g. `10001` matches both
    New York, US and Troyes, FR.

    `country_code` is an optional ISO-3166-1 alpha-2 hint ("US", "UA",
    "GB"): use it to disambiguate homonyms like "Moscow, RU" vs
    "Moscow, ID". The response includes `admin1` (state/region) and
    `postcodes` (list of zip codes tied to the hit) so the caller can
    verify the right place was picked.

    Ambiguous queries (e.g. "Springfield" without a country) still
    return a single top-ranked hit here — call `list_cities(...)` first
    if you want to see every candidate and pick one explicitly.
    """
    return await _geocode(city, country_code=country_code)


@mcp.tool()
async def list_cities(
    query: str,
    country_code: str | None = None,
    limit: int = 5,
) -> dict:
    """Return every geocoding candidate for an ambiguous city or postal code.

    Use this tool **before** the weather tools whenever the user's
    request is vague ("what's the weather in Springfield?" —
    Springfield exists in dozens of US states, plus MO, IL, etc.). The
    response contains a `candidates` list with country, admin1 (state
    or region), lat/lon, timezone and population; the caller should
    either pick one on the user's behalf or ask the user to clarify.

    `country_code` narrows the candidate pool server-side. `limit`
    caps the list length (1-10, default 5).
    """
    limit = max(1, min(int(limit), 10))
    async with httpx.AsyncClient(timeout=15.0) as client:
        r = await client.get(
            GEOCODE_URL,
            params={"name": query, "count": 10, "language": "en"},
        )
        r.raise_for_status()
        data = r.json()
    results = data.get("results") or []
    if country_code:
        cc_upper = country_code.strip().upper()
        results = [h for h in results if (h.get("country_code") or "").upper() == cc_upper]
    candidates = [
        {
            "name": h["name"],
            "country": h.get("country"),
            "country_code": h.get("country_code"),
            "admin1": h.get("admin1"),
            "latitude": h["latitude"],
            "longitude": h["longitude"],
            "timezone": h.get("timezone"),
            "population": h.get("population"),
            "postcodes": h.get("postcodes"),
        }
        for h in results[:limit]
    ]
    return {
        "query": query,
        "country_code": country_code,
        "candidates": candidates,
        "ambiguous": len(candidates) > 1,
    }


@mcp.tool()
async def get_today(timezone: str = "UTC") -> dict:
    """Return today's date, weekday and the timezone used for anchoring.

    Call this first when the user asks about "today", "tomorrow" or a
    weekday — the forecast tools return ISO dates, not relative labels,
    so the model needs an explicit anchor. `timezone` accepts IANA names
    like "Europe/Kyiv" or "UTC"; defaults to UTC when unset.
    """
    try:
        tz = ZoneInfo(timezone) if timezone else _tz.utc
    except Exception:
        tz = _tz.utc
        timezone = "UTC"
    now = datetime.now(tz)
    return {
        "date": now.date().isoformat(),
        "weekday": now.strftime("%A"),
        "iso_datetime": now.isoformat(timespec="seconds"),
        "timezone": timezone,
    }


@mcp.tool()
async def get_current_weather(city: str, country_code: str | None = None) -> dict:
    """Get the current weather for a city, postal code, or lat/lon name.

    `country_code` is an optional ISO-3166-1 alpha-2 hint to disambiguate
    homonyms (e.g. `Moscow, RU` vs `Moscow, ID`).
    """
    loc = await _geocode(city, country_code=country_code)
    async with httpx.AsyncClient(timeout=15.0) as client:
        r = await client.get(
            FORECAST_URL,
            params={
                "latitude": loc["latitude"],
                "longitude": loc["longitude"],
                "current": "temperature_2m,relative_humidity_2m,apparent_temperature,"
                           "precipitation,weather_code,wind_speed_10m,wind_direction_10m",
                "timezone": "auto",
            },
        )
        r.raise_for_status()
        data = r.json()
    cur = data.get("current", {})
    return {
        "location": f"{loc['name']}, {loc['country']}",
        "time": cur.get("time"),
        "temperature_c": cur.get("temperature_2m"),
        "apparent_temperature_c": cur.get("apparent_temperature"),
        "humidity_pct": cur.get("relative_humidity_2m"),
        "precipitation_mm": cur.get("precipitation"),
        "wind_kmh": cur.get("wind_speed_10m"),
        "wind_direction_deg": cur.get("wind_direction_10m"),
        "conditions": WEATHER_CODES.get(cur.get("weather_code"), "unknown"),
    }


@mcp.tool()
async def get_forecast(
    city: str,
    days: int = 7,
    country_code: str | None = None,
) -> dict:
    """Get a daily forecast (1-16 days) for a city or postal code.

    Each day entry includes `date` (ISO) plus `day_label` ("today",
    "tomorrow", "in N days") anchored to the city's local timezone —
    so the model does not need to know the current date to answer
    "what's tomorrow". `country_code` is an optional disambiguation
    hint; see `geocode_city` for details.
    """
    days = max(1, min(int(days), 16))
    loc = await _geocode(city, country_code=country_code)
    async with httpx.AsyncClient(timeout=15.0) as client:
        r = await client.get(
            FORECAST_URL,
            params={
                "latitude": loc["latitude"],
                "longitude": loc["longitude"],
                "daily": "weather_code,temperature_2m_max,temperature_2m_min,"
                         "precipitation_sum,precipitation_probability_max,wind_speed_10m_max",
                "forecast_days": days,
                "timezone": "auto",
            },
        )
        r.raise_for_status()
        data = r.json()
    d = data.get("daily", {})
    dates = d.get("time", [])
    # Anchor "today" to the city's own timezone, not the server's —
    # Open-Meteo returns daily rows in that timezone when
    # `timezone=auto`, so using UTC here would mislabel evenings where
    # the server is UTC+ but the city has already rolled the date.
    try:
        today = datetime.now(ZoneInfo(loc.get("timezone") or "UTC")).date()
    except Exception:
        today = datetime.now(_tz.utc).date()
    out = []
    for i, date_iso in enumerate(dates):
        out.append({
            "date": date_iso,
            "day_label": _day_label(date.fromisoformat(date_iso), today),
            "conditions": WEATHER_CODES.get(d["weather_code"][i], "unknown"),
            "temp_min_c": d["temperature_2m_min"][i],
            "temp_max_c": d["temperature_2m_max"][i],
            "precipitation_mm": d["precipitation_sum"][i],
            "precipitation_probability_pct": d["precipitation_probability_max"][i],
            "wind_max_kmh": d["wind_speed_10m_max"][i],
        })
    return {
        "location": f"{loc['name']}, {loc['country']}",
        "timezone": loc.get("timezone"),
        "days": out,
    }


def _run_http_with_auth() -> None:
    import uvicorn
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.responses import JSONResponse

    app = mcp.streamable_http_app()

    if AUTH_TOKEN:
        expected = f"Bearer {AUTH_TOKEN}"

        class BearerAuthMiddleware(BaseHTTPMiddleware):
            async def dispatch(self, request, call_next):
                if not secrets.compare_digest(
                    request.headers.get("authorization", ""), expected
                ):
                    return JSONResponse(
                        {"error": "unauthorized"},
                        status_code=401,
                        headers={"WWW-Authenticate": "Bearer"},
                    )
                return await call_next(request)

        app.add_middleware(BearerAuthMiddleware)

    uvicorn.run(app, host=HOST, port=PORT, log_level="info")


if __name__ == "__main__":
    if TRANSPORT == "streamable-http":
        _run_http_with_auth()
    else:
        mcp.run(transport=TRANSPORT)
