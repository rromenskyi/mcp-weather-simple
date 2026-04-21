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


# GeoNames feature_code → human label. Exposed to the caller so the
# model (or a client) can reason about what kind of place a candidate
# is — a town, a neighborhood, a mountain, a lake — and decide whether
# that matches the user's intent. A few codes are grouped under one
# bucket (e.g. every `PPL*` except `PPLX`/`PPLH`/`PPLQ`/`PPLW` is just
# "city") because the distinction rarely matters to weather queries.
_FEATURE_TYPE_BY_CODE = {
    "PPL": "city", "PPLA": "city", "PPLA2": "city", "PPLA3": "city",
    "PPLA4": "city", "PPLA5": "city", "PPLC": "city",
    "PPLCH": "historical place", "PPLH": "historical place",
    "PPLQ": "abandoned place", "PPLW": "destroyed place",
    "PPLL": "village", "PPLF": "farm", "PPLR": "religious place",
    "PPLS": "populated places", "PPLX": "neighborhood",
    "MT": "mountain", "MTS": "mountains", "PK": "peak", "PKS": "peaks",
    "HLL": "hill", "HLLS": "hills", "RDGE": "ridge", "CLF": "cliff",
    "LK": "lake", "LKS": "lakes", "RSV": "reservoir",
    "STM": "stream", "STMS": "streams", "RVN": "ravine",
    "ISL": "island", "ISLS": "islands", "ISLET": "islet",
    "PRK": "park", "RES": "reserve", "FRST": "forest",
    "BCH": "beach", "CAPE": "cape", "BAY": "bay", "COVE": "cove",
    "AIRP": "airport", "AIRF": "airfield",
    "RSTN": "train station", "BUSSTN": "bus station",
}


def _feature_type(code: str | None) -> str | None:
    if not code:
        return None
    return _FEATURE_TYPE_BY_CODE.get(code.upper(), code)  # fall back to the raw code


def _annotate(hit: dict) -> dict:
    """Normalize a geocoding hit into a caller-facing dict with human labels."""
    fc = hit.get("feature_code")
    return {
        "name": hit["name"],
        "country": hit.get("country"),
        "country_code": hit.get("country_code"),
        "admin1": hit.get("admin1"),
        "latitude": hit["latitude"],
        "longitude": hit["longitude"],
        "timezone": hit.get("timezone"),
        "population": hit.get("population"),
        "postcodes": hit.get("postcodes"),
        "feature_code": fc,
        "feature_type": _feature_type(fc),
    }


async def _geocode(city: str, country_code: str | None = None) -> dict:
    # Pull a few candidates so we can filter by `country_code` client-side —
    # Open-Meteo's geocoding endpoint does not accept a country filter and
    # returns towns mixed with mountains, lakes, neighborhoods and islands
    # bearing the same name. We intentionally do NOT silently exclude
    # non-city hits here — a caller that wants only populated places can
    # use `list_places(feature_types=["city"])` and pick explicitly, while
    # someone asking about weather on Mt. Everest or in the Bountiful
    # Islands should still get a useful top hit back.
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
    return _annotate(results[0])


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

    Ambiguous queries (e.g. "Springfield" without a country, or a
    query that could resolve to a town OR a mountain of the same name)
    still return a single top-ranked hit here — call `list_places(...)`
    first if you want to see every candidate, filter by feature type,
    and pick one explicitly.
    """
    return await _geocode(city, country_code=country_code)


@mcp.tool()
async def list_places(
    query: str,
    country_code: str | None = None,
    feature_types: list[str] | None = None,
    limit: int = 5,
) -> dict:
    """Return every geocoding candidate for an ambiguous query.

    Deliberately general-purpose: the Open-Meteo geocoder returns not
    just towns but also mountains, lakes, parks, islands, neighborhoods
    and airports bearing the same name. Each candidate carries a
    `feature_type` human label ("city", "mountain", "lake",
    "neighborhood", "park", "peak", "island", "airport", …) so the
    caller can match the user's intent ("weather on Mt. Bountiful" vs
    "weather in Bountiful" are legitimately different places).

    Parameters:
    - `query`: name or postal code. Same semantics as `geocode_city`.
    - `country_code`: ISO-3166-1 alpha-2 hint ("US", "UA"). Narrows the
      candidate pool server-side.
    - `feature_types`: optional allowlist of human labels
      (e.g. `["city", "village"]` for only populated places, or
      `["mountain", "peak", "hill"]` for only high ground). Pass
      `None` / empty to keep every feature type in the results.
    - `limit`: caps the list length (1-10, default 5).

    Use this tool whenever the user's request is vague — either because
    the name is ambiguous ("Springfield"), the intent isn't clear
    (town vs mountain vs lake), or the first hit's `feature_type`
    doesn't match what the user seems to want. The caller can then
    surface the disambiguation choice to the user or pick one on their
    behalf.
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
    if feature_types:
        allowed = {ft.strip().lower() for ft in feature_types if ft}
        results = [h for h in results if (_feature_type(h.get("feature_code")) or "").lower() in allowed]
    candidates = [_annotate(h) for h in results[:limit]]
    return {
        "query": query,
        "country_code": country_code,
        "feature_types": feature_types,
        "candidates": candidates,
        "ambiguous": len(candidates) > 1,
    }


@mcp.tool()
async def get_local_time(city: str, country_code: str | None = None) -> dict:
    """Return the current local date, time and timezone of a city or zipcode.

    Answers "what time is it in Kyiv?" without the model having to know
    the city's offset. Internally calls the same geocoder the weather
    tools use — so `country_code` is an optional ISO-3166-1 alpha-2
    hint and the response's `name` / `country` / `admin1` confirm which
    place was picked.
    """
    loc = await _geocode(city, country_code=country_code)
    try:
        tz = ZoneInfo(loc.get("timezone") or "UTC")
    except Exception:
        tz = _tz.utc
    now = datetime.now(tz)
    return {
        "location": f"{loc['name']}, {loc.get('country', '')}".rstrip(", "),
        "admin1": loc.get("admin1"),
        "country_code": loc.get("country_code"),
        "timezone": loc.get("timezone"),
        "date": now.date().isoformat(),
        "time": now.strftime("%H:%M:%S"),
        "weekday": now.strftime("%A"),
        "iso_datetime": now.isoformat(timespec="seconds"),
        "utc_offset": now.strftime("%z"),
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
