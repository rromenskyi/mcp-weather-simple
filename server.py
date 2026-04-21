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
# Archive-api covers 1940-present. Same query grammar as /forecast but
# requires `start_date` / `end_date` instead of `forecast_days`.
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
# Free, no-auth air-quality endpoint. Returns PM2.5 / PM10 / ozone /
# nitrogen-dioxide / sulphur-dioxide / carbon-monoxide + European and
# US AQI indices.
AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
# Free, no-auth, HTTPS GeoIP. Accepts an IP in the path or `/` for
# auto-detection based on the caller. Returns ip, city, region,
# country_code, latitude, longitude and timezone.id (IANA name). We
# compute the local clock ourselves from timezone.id — ipwho.is'
# `timezone.current_time` is occasionally null (seen on IPv6 lookups).
GEOIP_URL = "https://ipwho.is"

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
# that matches the user's intent.
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
    # use `search_places(feature_types=["city"])` and pick explicitly, while
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
async def find_place_coordinates(city: str, country_code: str | None = None) -> dict:
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
    still return a single top-ranked hit here — call `search_places(...)`
    first if you want to see every candidate, filter by feature type,
    and pick one explicitly.
    """
    return await _geocode(city, country_code=country_code)


@mcp.tool()
async def search_places(
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
    - `query`: name or postal code. Same semantics as `find_place_coordinates`.
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


# ── No-arg "user-question" shortcuts ───────────────────────────────────────
#
# Pre-composed tools whose NAME alone tells a weak model which user question
# they answer. Each one auto-detects the caller's city via GeoIP (so the
# user doesn't have to name their city) and then calls the appropriate
# lower-level tool. The lower-level tools remain available for precise
# follow-ups — these are just the obvious defaults.


async def _here_city() -> tuple[str, str | None]:
    """Resolve the caller's city + country_code from its public IP."""
    loc = await detect_my_location_by_ip()
    city = loc.get("city")
    cc = loc.get("country_code")
    if not city:
        raise ValueError(
            "GeoIP lookup did not resolve a city for the caller's public IP. "
            "Ask the user to name a city and call the explicit tool instead."
        )
    return city, cc


@mcp.tool()
async def get_weather_outside_right_now() -> dict:
    """What's the weather outside right now, where the user is.

    Zero-argument shortcut for "какая погода на улице?" / "what's the
    weather outside?". Auto-detects the user's city via GeoIP and
    returns the current conditions — temperature, humidity, wind,
    precipitation. If the user is asking about a specific city by
    name, call `get_current_weather_in_city(city)` instead.
    """
    city, cc = await _here_city()
    weather = await get_current_weather_in_city(city, country_code=cc)
    weather["_note"] = "location auto-detected from caller IP"
    return weather


@mcp.tool()
async def get_weather_forecast_for_today() -> dict:
    """Today's forecast (high / low / conditions / rain chance) where the user is.

    Zero-argument shortcut for "какая погода сегодня?" / "what's the
    weather today?". Auto-detects the user's city via GeoIP and
    returns ONE entry from the daily forecast labelled "today".
    """
    city, cc = await _here_city()
    result = await get_weather_forecast(city, days=1, country_code=cc)
    result["_note"] = "location auto-detected from caller IP"
    return result


@mcp.tool()
async def get_current_time_where_i_am() -> dict:
    """What time / date / timezone it is right now, where the user is.

    Zero-argument shortcut for "который сейчас час?" / "what time
    is it?" / "what's today's date?". Auto-detects the user's city
    and timezone via GeoIP. Returns local time, local date, weekday,
    UTC offset and the IANA timezone name. If the user is asking
    about a specific city instead of themselves, call
    `get_current_time_in_city(city)`.
    """
    loc = await detect_my_location_by_ip()
    return {
        "city": loc.get("city"),
        "country": loc.get("country"),
        "country_code": loc.get("country_code"),
        "timezone": loc.get("timezone"),
        "local_date": loc.get("local_date"),
        "local_time": loc.get("local_time"),
        "weekday": loc.get("weekday"),
        "utc_offset": loc.get("utc_offset"),
        "iso_datetime": loc.get("iso_datetime"),
        "_note": "location and timezone auto-detected from caller IP",
    }


@mcp.tool()
async def get_weather_forecast_for_tomorrow() -> dict:
    """Tomorrow's forecast (high / low / conditions / rain chance) where the user is.

    Zero-argument shortcut for "какая погода завтра?" / "what's the
    forecast for tomorrow?". Auto-detects the user's city via GeoIP
    and returns TWO entries from the daily forecast (today + tomorrow,
    so the model can compare); `day_label` on each says which is which.
    """
    city, cc = await _here_city()
    result = await get_weather_forecast(city, days=2, country_code=cc)
    result["_note"] = "location auto-detected from caller IP; includes today and tomorrow"
    return result


@mcp.tool()
async def detect_my_location_by_ip(ip: str | None = None) -> dict:
    """Detect the caller's approximate location from its public IP.

    Call this tool when the user asks "where am I?", "what's the
    weather here?", "what time is it here?" — i.e. wants a
    location-aware answer without naming a city. Backed by the public
    HTTPS service ipwho.is (no API key, no local GeoIP database).

    Parameters:
    - `ip`: an explicit IPv4/IPv6 to look up. When `None` (default)
      the service auto-detects the IP the HTTP request arrives from.

    Returns `city`, `region`, `country`, `country_code`, `latitude`,
    `longitude`, `timezone` (IANA id), `local_time`, `weekday` and
    `utc_offset`. Feed `city` (plus `country_code` for disambiguation)
    straight into `get_current_weather_in_city` / `get_weather_forecast` to answer
    "weather here".

    Limitation: when the MCP server runs inside a Kubernetes cluster
    or behind any NAT/VPN, the auto-detected IP is the cluster / NAT
    gateway's egress IP, not the end user's browser IP — the reported
    city is where the server's uplink terminates, which may be far
    from the user. Pass `ip` explicitly when that matters.
    """
    target = f"{GEOIP_URL}/{ip}" if ip else f"{GEOIP_URL}/"
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.get(target)
        r.raise_for_status()
        data = r.json()
    if not data.get("success", True):
        raise ValueError(f"GeoIP lookup failed: {data.get('message', 'unknown reason')}")
    tz_id = (data.get("timezone") or {}).get("id") or "UTC"
    try:
        tz = ZoneInfo(tz_id)
    except Exception:
        tz = _tz.utc
        tz_id = "UTC"
    now = datetime.now(tz)
    return {
        "ip": data.get("ip"),
        "city": data.get("city"),
        "region": data.get("region"),
        "country": data.get("country"),
        "country_code": data.get("country_code"),
        "latitude": data.get("latitude"),
        "longitude": data.get("longitude"),
        "timezone": tz_id,
        "local_time": now.strftime("%H:%M:%S"),
        "local_date": now.date().isoformat(),
        "weekday": now.strftime("%A"),
        "iso_datetime": now.isoformat(timespec="seconds"),
        "utc_offset": now.strftime("%z"),
    }


@mcp.tool()
async def get_current_time_in_city(city: str, country_code: str | None = None) -> dict:
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
async def get_current_date(timezone: str = "UTC") -> dict:
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
async def get_current_weather_in_city(city: str, country_code: str | None = None) -> dict:
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
async def get_weather_forecast(
    city: str,
    days: int = 7,
    country_code: str | None = None,
) -> dict:
    """Get a daily forecast (1-16 days) for a city or postal code.

    Each day entry includes `date` (ISO) plus `day_label` ("today",
    "tomorrow", "in N days") anchored to the city's local timezone —
    so the model does not need to know the current date to answer
    "what's tomorrow". `country_code` is an optional disambiguation
    hint; see `find_place_coordinates` for details.
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


@mcp.tool()
async def get_hourly_forecast(
    city: str,
    hours: int = 24,
    country_code: str | None = None,
) -> dict:
    """Get an hour-by-hour forecast (1-168 hours) for a city or zipcode.

    Answers "will it rain this afternoon?", "when does the storm hit
    tomorrow morning?" — questions where daily aggregates hide the
    timing. Each hour entry is anchored to the city's local timezone.
    `hours` caps the output length (max 168 = 7 days ahead).
    """
    hours = max(1, min(int(hours), 168))
    # `forecast_days` controls how many days Open-Meteo computes;
    # round up from the requested hours so we always have enough rows.
    days = max(1, min((hours + 23) // 24, 7))
    loc = await _geocode(city, country_code=country_code)
    async with httpx.AsyncClient(timeout=15.0) as client:
        r = await client.get(
            FORECAST_URL,
            params={
                "latitude": loc["latitude"],
                "longitude": loc["longitude"],
                "hourly": "temperature_2m,relative_humidity_2m,precipitation,"
                          "precipitation_probability,weather_code,cloud_cover,"
                          "wind_speed_10m,wind_direction_10m",
                "forecast_days": days,
                "timezone": "auto",
            },
        )
        r.raise_for_status()
        data = r.json()
    h = data.get("hourly", {})
    times = h.get("time", [])
    out = []
    for i, t in enumerate(times[:hours]):
        out.append({
            "time": t,  # ISO local timestamp in the city's timezone
            "temperature_c": h["temperature_2m"][i],
            "humidity_pct": h["relative_humidity_2m"][i],
            "precipitation_mm": h["precipitation"][i],
            "precipitation_probability_pct": h["precipitation_probability"][i],
            "cloud_cover_pct": h["cloud_cover"][i],
            "conditions": WEATHER_CODES.get(h["weather_code"][i], "unknown"),
            "wind_kmh": h["wind_speed_10m"][i],
            "wind_direction_deg": h["wind_direction_10m"][i],
        })
    return {
        "location": f"{loc['name']}, {loc['country']}",
        "timezone": loc.get("timezone"),
        "hours": out,
    }


@mcp.tool()
async def get_sunrise_sunset(
    city: str,
    date_iso: str | None = None,
    days: int = 1,
    country_code: str | None = None,
) -> dict:
    """Sunrise / sunset / daylight duration for a city for one or more days.

    Answers "when does the sun set in Reykjavik today?", "what's the
    daylight duration on June 21 in Kyiv?". Defaults to today (1 day).

    - `date_iso`: optional anchor (YYYY-MM-DD). When set, the window
      starts on that date. When `None`, starts today in the city's
      local timezone.
    - `days`: number of consecutive days to return (1-16).
    """
    days = max(1, min(int(days), 16))
    loc = await _geocode(city, country_code=country_code)
    params: dict[str, str | int | float] = {
        "latitude": loc["latitude"],
        "longitude": loc["longitude"],
        "daily": "sunrise,sunset,daylight_duration,sunshine_duration",
        "timezone": "auto",
    }
    if date_iso:
        start = date.fromisoformat(date_iso)
        end = date.fromordinal(start.toordinal() + days - 1)
        params["start_date"] = start.isoformat()
        params["end_date"] = end.isoformat()
    else:
        params["forecast_days"] = days
    async with httpx.AsyncClient(timeout=15.0) as client:
        r = await client.get(FORECAST_URL, params=params)
        r.raise_for_status()
        data = r.json()
    d = data.get("daily", {})
    out = []
    for i, day_iso in enumerate(d.get("time", [])):
        # `daylight_duration` and `sunshine_duration` come in seconds.
        dl_s = d.get("daylight_duration", [None])[i] or 0
        ss_s = d.get("sunshine_duration", [None])[i] or 0
        out.append({
            "date": day_iso,
            "sunrise": d.get("sunrise", [None])[i],
            "sunset": d.get("sunset", [None])[i],
            "daylight_duration_hhmm": f"{int(dl_s // 3600):02d}:{int((dl_s % 3600) // 60):02d}",
            "sunshine_duration_hhmm": f"{int(ss_s // 3600):02d}:{int((ss_s % 3600) // 60):02d}",
        })
    return {
        "location": f"{loc['name']}, {loc['country']}",
        "timezone": loc.get("timezone"),
        "days": out,
    }


@mcp.tool()
async def get_air_quality(city: str, country_code: str | None = None) -> dict:
    """Current air quality (PM2.5, PM10, ozone, NO2, SO2, CO + AQI) in a city.

    Uses Open-Meteo's air-quality endpoint (no key required). Returns
    both the European AQI and the US AQI scales — the model can pick
    the right one for the user's region. Answers "is the air safe in
    Delhi today?", "should I wear a mask outside?".
    """
    loc = await _geocode(city, country_code=country_code)
    async with httpx.AsyncClient(timeout=15.0) as client:
        r = await client.get(
            AIR_QUALITY_URL,
            params={
                "latitude": loc["latitude"],
                "longitude": loc["longitude"],
                "current": "european_aqi,us_aqi,pm10,pm2_5,carbon_monoxide,"
                           "nitrogen_dioxide,sulphur_dioxide,ozone",
                "timezone": "auto",
            },
        )
        r.raise_for_status()
        data = r.json()
    cur = data.get("current", {})
    return {
        "location": f"{loc['name']}, {loc['country']}",
        "timezone": loc.get("timezone"),
        "time": cur.get("time"),
        "european_aqi": cur.get("european_aqi"),
        "us_aqi": cur.get("us_aqi"),
        "pm2_5_ugm3": cur.get("pm2_5"),
        "pm10_ugm3": cur.get("pm10"),
        "ozone_ugm3": cur.get("ozone"),
        "nitrogen_dioxide_ugm3": cur.get("nitrogen_dioxide"),
        "sulphur_dioxide_ugm3": cur.get("sulphur_dioxide"),
        "carbon_monoxide_ugm3": cur.get("carbon_monoxide"),
    }


@mcp.tool()
async def get_weather_by_coordinates(latitude: float, longitude: float) -> dict:
    """Current weather at raw lat/lon — skips the geocoder entirely.

    Use when the user already has coordinates (e.g. from
    `detect_my_location_by_ip` or pasted from a map app), or the
    location isn't a named place (lake, trailhead, offshore). No city
    lookup, no homonym disambiguation — just the weather at that point.
    """
    async with httpx.AsyncClient(timeout=15.0) as client:
        r = await client.get(
            FORECAST_URL,
            params={
                "latitude": latitude,
                "longitude": longitude,
                "current": "temperature_2m,relative_humidity_2m,apparent_temperature,"
                           "precipitation,weather_code,wind_speed_10m,wind_direction_10m",
                "timezone": "auto",
            },
        )
        r.raise_for_status()
        data = r.json()
    cur = data.get("current", {})
    return {
        "latitude": latitude,
        "longitude": longitude,
        "timezone": data.get("timezone"),
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
async def get_historical_weather(
    city: str,
    start_date_iso: str,
    end_date_iso: str | None = None,
    country_code: str | None = None,
) -> dict:
    """Daily weather from the Open-Meteo archive (1940-present).

    Answers "how was the weather on my birthday in 1998?", "was it
    unusually cold in Kyiv last February?". `start_date_iso` is
    required (YYYY-MM-DD); `end_date_iso` defaults to `start_date_iso`
    for a single-day lookup. Max 31 days per request.
    """
    start = date.fromisoformat(start_date_iso)
    end = date.fromisoformat(end_date_iso) if end_date_iso else start
    if end < start:
        raise ValueError(f"end_date_iso ({end}) precedes start_date_iso ({start})")
    span_days = (end - start).days + 1
    if span_days > 31:
        raise ValueError(
            f"Date span too large ({span_days} days). Split into chunks of 31 days or fewer."
        )
    loc = await _geocode(city, country_code=country_code)
    async with httpx.AsyncClient(timeout=20.0) as client:
        r = await client.get(
            ARCHIVE_URL,
            params={
                "latitude": loc["latitude"],
                "longitude": loc["longitude"],
                "start_date": start.isoformat(),
                "end_date": end.isoformat(),
                "daily": "weather_code,temperature_2m_max,temperature_2m_min,"
                         "temperature_2m_mean,precipitation_sum,"
                         "wind_speed_10m_max",
                "timezone": "auto",
            },
        )
        r.raise_for_status()
        data = r.json()
    d = data.get("daily", {})
    out = []
    for i, day_iso in enumerate(d.get("time", [])):
        out.append({
            "date": day_iso,
            "conditions": WEATHER_CODES.get(d["weather_code"][i], "unknown"),
            "temp_min_c": d["temperature_2m_min"][i],
            "temp_max_c": d["temperature_2m_max"][i],
            "temp_mean_c": d["temperature_2m_mean"][i],
            "precipitation_mm": d["precipitation_sum"][i],
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
