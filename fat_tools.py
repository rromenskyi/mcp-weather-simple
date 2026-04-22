"""Fat domain-tools for `MCP_ROUTER_MODE=fat_tools`.

Collapses the 23 narrow `@mcp.tool`s in ``server.py`` into 4 fat tools —
``weather`` / ``geo`` / ``knowledge`` / ``radio`` — where the model picks
a domain at the top level and passes an ``action`` discriminator to
select the concrete operation inside. Goal is to shrink the per-turn
tool catalog from ~4500 tokens to ~1500-2000 so that small models on
CPU (i7 + qwen3.5:9b) can prefill the catalog in acceptable time.

**Not auto-loaded** — ``server._install_router()`` calls
``install_fat_tools(mcp)`` only when ``MCP_ROUTER_MODE=fat_tools``. The
narrow `@mcp.tool`s remain registered; ``list_tools`` override in
``server.py`` hides them when this mode is active. That keeps the
default path bit-identical to v1.0.0 and the fat surface fully
reversible.

Tradeoff, for the record: small models generally pick well from a
short list of distinctly-named tools, but they're less reliable at
routing inside a fat tool via an ``action`` enum. This module is a
test — eval numbers on qwen3.5:9b vs the monolith baseline will tell
us whether the latency win outweighs the tool-picking quality drop.
"""

from __future__ import annotations

from typing import Literal

import server


# ── weather ─────────────────────────────────────────────────────────────

_WEATHER_ACTIONS = Literal[
    "current_here",
    "today_here",
    "tomorrow_here",
    "current_in_city",
    "forecast_days",
    "hourly",
    "sunrise_sunset",
    "air_quality",
    "by_coordinates",
    "historical",
]


async def weather(
    action: _WEATHER_ACTIONS,
    city: str | None = None,
    country_code: str | None = None,
    latitude: float | None = None,
    longitude: float | None = None,
    days: int = 7,
    hours: int = 24,
    date_iso: str | None = None,
    start_date_iso: str | None = None,
    end_date_iso: str | None = None,
) -> dict:
    """Weather, air quality, sunrise/sunset. Pick `action` to select the operation.

    Actions (each needs only the listed parameters):
      - `current_here` — current weather at user's detected location. No args.
      - `today_here` — today's forecast at user's location. No args.
      - `tomorrow_here` — tomorrow's forecast at user's location. No args.
      - `current_in_city` — current weather in a named place.
          Needs: `city`. Optional: `country_code` (ISO-3166-1 alpha-2).
      - `forecast_days` — N-day forecast for a city.
          Needs: `city`, `days` (1-14, default 7). Optional: `country_code`.
      - `hourly` — hour-by-hour forecast for a city.
          Needs: `city`, `hours` (1-168, default 24). Optional: `country_code`.
      - `sunrise_sunset` — sunrise/sunset times for a city.
          Needs: `city`. Optional: `date_iso` (YYYY-MM-DD, default today),
          `days` (1-7, default 1), `country_code`.
      - `air_quality` — PM2.5/PM10/O3/NO2/SO2/CO + AQI for a city.
          Needs: `city`. Optional: `country_code`.
      - `by_coordinates` — current weather at raw lat/lon.
          Needs: `latitude`, `longitude`. **Do not invent coordinates.**
      - `historical` — daily weather on a past date range.
          Needs: `city`, `start_date_iso` (YYYY-MM-DD).
          Optional: `end_date_iso`, `country_code`.

    `city` is ALWAYS a single token: a place name (`"Kyiv"`), a postal
    code (`"84010"`), or the shape `"City, Region"` when disambiguation
    is required. Full-sentence queries will fail.
    """
    if action == "current_here":
        return await server.get_weather_outside_right_now()
    if action == "today_here":
        return await server.get_weather_forecast_for_today()
    if action == "tomorrow_here":
        return await server.get_weather_forecast_for_tomorrow()
    if action == "current_in_city":
        _require(city=city)
        return await server.get_current_weather_in_city(city, country_code)
    if action == "forecast_days":
        _require(city=city)
        return await server.get_weather_forecast(city, days, country_code)
    if action == "hourly":
        _require(city=city)
        return await server.get_hourly_forecast(city, hours, country_code)
    if action == "sunrise_sunset":
        _require(city=city)
        return await server.get_sunrise_sunset(city, date_iso, days, country_code)
    if action == "air_quality":
        _require(city=city)
        return await server.get_air_quality(city, country_code)
    if action == "by_coordinates":
        _require(latitude=latitude, longitude=longitude)
        return await server.get_weather_by_coordinates(latitude, longitude)
    if action == "historical":
        _require(city=city, start_date_iso=start_date_iso)
        return await server.get_historical_weather(
            city, start_date_iso, end_date_iso, country_code
        )
    raise ValueError(f"weather: unknown action {action!r}")


# ── geo ─────────────────────────────────────────────────────────────────

_GEO_ACTIONS = Literal[
    "find_coordinates",
    "search_places",
    "resolve_address",
    "detect_my_location",
    "lookup_ip",
    "time_here",
    "time_in_city",
    "date_in_timezone",
]


async def geo(
    action: _GEO_ACTIONS,
    city: str | None = None,
    country_code: str | None = None,
    query: str | None = None,
    address: str | None = None,
    ip: str | None = None,
    timezone: str = "UTC",
    feature_types: list[str] | None = None,
    limit: int = 5,
) -> dict:
    """Geocoding, reverse-geocoding, IP geolocation, time, and date.

    Actions:
      - `find_coordinates` — lat/lon + canonical name for a single known
          place. Needs: `city`. Optional: `country_code`.
      - `search_places` — list candidates for an ambiguous place query
          (town / mountain / airport / etc, each tagged by feature_type).
          Needs: `query`. Optional: `country_code`, `feature_types`, `limit`.
      - `resolve_address` — parse a free-form street address into
          `{city, region, country, coordinates}`.
          Needs: `address`. Optional: `country_code`.
      - `detect_my_location` — where the user is, inferred from their
          public IP (server-side). No args.
      - `lookup_ip` — geolocate an arbitrary IP address.
          Needs: `ip`.
      - `time_here` — current local time at user's detected location.
          No args.
      - `time_in_city` — current local time in a named city.
          Needs: `city`. Optional: `country_code`.
      - `date_in_timezone` — today's date in a named timezone.
          Optional: `timezone` (IANA, default `"UTC"`).
    """
    if action == "find_coordinates":
        _require(city=city)
        return await server.find_place_coordinates(city, country_code)
    if action == "search_places":
        _require(query=query)
        return await server.search_places(query, country_code, feature_types, limit)
    if action == "resolve_address":
        _require(address=address)
        return await server.resolve_address(address, country_code)
    if action == "detect_my_location":
        return await server.detect_my_location_by_ip()
    if action == "lookup_ip":
        _require(ip=ip)
        return await server.lookup_ip_geolocation(ip)
    if action == "time_here":
        return await server.get_current_time_where_i_am()
    if action == "time_in_city":
        _require(city=city)
        return await server.get_current_time_in_city(city, country_code)
    if action == "date_in_timezone":
        return await server.get_current_date(timezone)
    raise ValueError(f"geo: unknown action {action!r}")


# ── knowledge ───────────────────────────────────────────────────────────

_KNOWLEDGE_ACTIONS = Literal[
    "wikipedia",
    "country_info",
    "public_holidays",
    "convert_currency",
]


async def knowledge(
    action: _KNOWLEDGE_ACTIONS,
    title: str | None = None,
    lang: str = "en",
    country: str | None = None,
    country_code: str | None = None,
    year: int | None = None,
    amount: float | None = None,
    from_currency: str | None = None,
    to_currency: str | None = None,
) -> dict:
    """Wikipedia, country facts, public-holiday calendar, currency conversion.

    Actions:
      - `wikipedia` — concise summary of a Wikipedia article.
          Needs: `title`. Optional: `lang` (ISO-639 code, default `"en"`).
      - `country_info` — official-name / capital / population / area / flag
          for a country.
          Needs: `country` (full name or ISO code).
      - `public_holidays` — government-recognised public holidays for a
          country in a given year.
          Needs: `country_code` (ISO-3166-1 alpha-2). Optional: `year`
          (default current year).
      - `convert_currency` — convert between currencies at today's ECB rate.
          Needs: `amount`, `from_currency` (ISO-4217), `to_currency` (ISO-4217).
    """
    if action == "wikipedia":
        _require(title=title)
        return await server.get_wikipedia_summary(title, lang)
    if action == "country_info":
        _require(country=country)
        return await server.get_country_info(country)
    if action == "public_holidays":
        _require(country_code=country_code)
        return await server.get_public_holidays(country_code, year)
    if action == "convert_currency":
        _require(
            amount=amount, from_currency=from_currency, to_currency=to_currency
        )
        return await server.convert_currency(amount, from_currency, to_currency)
    raise ValueError(f"knowledge: unknown action {action!r}")


# ── radio ───────────────────────────────────────────────────────────────


async def radio(
    country: str | None = None,
    tag: str | None = None,
    language: str | None = None,
    limit: int = 5,
) -> dict:
    """Find internet-radio stations (streams, not FM/AM frequencies).

    Pass at least one filter: `country` (ISO-2 like `"UA"` preferred, or
    full English name), `tag` (genre keyword: `"jazz"`, `"news"`,
    `"chillout"`), or `language` (full English name like `"russian"`,
    not the ISO code). `limit` caps the result count, 1-20 (default 5).
    """
    return await server.list_radio_stations(country, tag, language, limit)


# ── install ─────────────────────────────────────────────────────────────


def install_fat_tools(mcp) -> None:
    """Register the four fat-domain tools on the FastMCP instance.

    Called from ``server._install_router()`` only when router mode is
    ``fat_tools``. The narrow `@mcp.tool`s in ``server.py`` remain
    registered but are hidden by the ``list_tools`` override so the
    client only sees the fat surface.
    """
    mcp.tool()(weather)
    mcp.tool()(geo)
    mcp.tool()(knowledge)
    mcp.tool()(radio)


# ── helpers ─────────────────────────────────────────────────────────────


def _require(**kwargs) -> None:
    """Raise ValueError if any kwarg is None / empty.

    Keeps the dispatcher's per-action argument requirements enforceable
    without duplicating `if x is None: raise` boilerplate for every
    action. Error message lists all missing fields so the model can fix
    them in one retry.
    """
    missing = [k for k, v in kwargs.items() if v in (None, "", [])]
    if missing:
        raise ValueError(
            f"missing required argument(s): {', '.join(missing)}"
        )
