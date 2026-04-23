"""Ultra-lean variant of `fat_tools.py` for `MCP_ROUTER_MODE=fat_tools_lean`.

Same four domain-tools (weather / geo / knowledge / radio), same
action names, but the signature drops from `(action, name=None,
lat=None, lon=None, ...)` down to just `(action, params={})`. That's
a huge win on FastMCP-generated schema: every optional kwarg in
fat_tools.py expands to `anyOf: [{type: string}, {type: null}]` plus
a `title` block — roughly 150 tokens per tool purely of pydantic
nullability noise. Collapsing to one freeform `params` dict cuts
that out entirely.

Measured on 2026-04-22 against fat_tools baseline of ~1885 tokens:
lean variant lands around **~900-1000 tokens** total catalog
(~-50% vs fat_tools, ~-80% vs monolith).

Trade-off: loss of parameter type hints in the schema — the model
only sees `params` as a plain object. The docstring is what teaches
it which keys each action needs. If hit rate drops noticeably vs
fat_tools, the schema was doing real work and we back out.

Action names are identical to `fat_tools.py`, so the eval scorer's
`NARROW_TO_FAT` canonicalisation works unchanged — cases.yaml
doesn't need any A/B-specific branching.

Uses lazy `import server` for the same circular-import reason as
fat_tools.py (see that module's docstring).
"""

from __future__ import annotations

from typing import Literal


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


async def weather(action: _WEATHER_ACTIONS, params: dict | None = None) -> dict:
    """Weather, air quality, sunrise/sunset. Pass `action` + `params` dict.

    Actions and their params:
      - `current_here`:     {} — current weather at user's detected location.
      - `today_here`:       {} — today's forecast at user's location.
      - `tomorrow_here`:    {} — tomorrow's forecast at user's location.
      - `current_in_city`:  {"city": str, "country_code"?: str}.
      - `forecast_days`:    {"city": str, "days"?: int=7, "country_code"?: str}.
      - `hourly`:           {"city": str, "hours"?: int=24, "country_code"?: str}.
      - `sunrise_sunset`:   {"city": str, "date_iso"?: str, "days"?: int=1,
                             "country_code"?: str}.
      - `air_quality`:      {"city": str, "country_code"?: str}.
      - `by_coordinates`:   {"latitude": float, "longitude": float}
                            — **do not invent coordinates**.
      - `historical`:       {"city": str, "start_date_iso": str,
                             "end_date_iso"?: str, "country_code"?: str}.

    `city` is always a single token (place name, postal code).
    """
    import server
    p = params or {}
    if action == "current_here":
        return await server.get_weather_outside_right_now()
    if action == "today_here":
        return await server.get_weather_forecast_for_today()
    if action == "tomorrow_here":
        return await server.get_weather_forecast_for_tomorrow()
    if action == "current_in_city":
        _require(p, "city")
        return await server.get_current_weather_in_city(p["city"], p.get("country_code"))
    if action == "forecast_days":
        _require(p, "city")
        return await server.get_weather_forecast(p["city"], p.get("days", 7), p.get("country_code"))
    if action == "hourly":
        _require(p, "city")
        return await server.get_hourly_forecast(p["city"], p.get("hours", 24), p.get("country_code"))
    if action == "sunrise_sunset":
        _require(p, "city")
        return await server.get_sunrise_sunset(
            p["city"], p.get("date_iso"), p.get("days", 1), p.get("country_code")
        )
    if action == "air_quality":
        _require(p, "city")
        return await server.get_air_quality(p["city"], p.get("country_code"))
    if action == "by_coordinates":
        _require(p, "latitude", "longitude")
        return await server.get_weather_by_coordinates(p["latitude"], p["longitude"])
    if action == "historical":
        _require(p, "city", "start_date_iso")
        return await server.get_historical_weather(
            p["city"], p["start_date_iso"], p.get("end_date_iso"), p.get("country_code")
        )
    raise ValueError(f"weather: unknown action {action!r}")


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


async def geo(action: _GEO_ACTIONS, params: dict | None = None) -> dict:
    """Geocoding / reverse-geocoding / IP geolocation / time / date.

    Actions and their params:
      - `find_coordinates`:   {"city": str, "country_code"?: str}.
      - `search_places`:      {"query": str, "country_code"?: str,
                               "feature_types"?: list[str], "limit"?: int=5}
                              — for ambiguous place queries.
      - `resolve_address`:    {"address": str, "country_code"?: str}
                              — free-form street address.
      - `detect_my_location`: {} — user's location via IP.
      - `lookup_ip`:          {"ip": str}.
      - `time_here`:          {} — user's local time.
      - `time_in_city`:       {"city": str, "country_code"?: str}.
      - `date_in_timezone`:   {"timezone"?: str="UTC"} — IANA tz name.
    """
    import server
    p = params or {}
    if action == "find_coordinates":
        _require(p, "city")
        return await server.find_place_coordinates(p["city"], p.get("country_code"))
    if action == "search_places":
        _require(p, "query")
        return await server.search_places(
            p["query"], p.get("country_code"), p.get("feature_types"), p.get("limit", 5)
        )
    if action == "resolve_address":
        _require(p, "address")
        return await server.resolve_address(p["address"], p.get("country_code"))
    if action == "detect_my_location":
        return await server.detect_my_location_by_ip()
    if action == "lookup_ip":
        _require(p, "ip")
        return await server.lookup_ip_geolocation(p["ip"])
    if action == "time_here":
        return await server.get_current_time_where_i_am()
    if action == "time_in_city":
        _require(p, "city")
        return await server.get_current_time_in_city(p["city"], p.get("country_code"))
    if action == "date_in_timezone":
        return await server.get_current_date(p.get("timezone", "UTC"))
    raise ValueError(f"geo: unknown action {action!r}")


_KNOWLEDGE_ACTIONS = Literal[
    "wikipedia",
    "country_info",
    "public_holidays",
    "convert_currency",
    "calculate",
]


async def knowledge(
    action: _KNOWLEDGE_ACTIONS, params: dict | None = None
) -> dict:
    """Wikipedia / country facts / public holidays / currency / arithmetic.

    Actions and their params:
      - `wikipedia`:         {"title": str, "lang"?: str="en"} — ISO-639 lang code.
      - `country_info`:      {"country": str} — full name or ISO code.
      - `public_holidays`:   {"country_code": str, "year"?: int=current} — ISO-3166-1 alpha-2.
      - `convert_currency`:  {"amount": float, "from_currency": str, "to_currency": str}
                              — ISO-4217 currency codes.
      - `calculate`:         {"expression": str} — AST-safe arithmetic, not `eval`.
                              **Use for ANY math** — model arithmetic unreliable on 4+
                              digits and chains. Geometry via explicit formulas
                              (`pi*r**2`, `hypot(a,b)`, `(4/3)*pi*r**3`). Supports
                              sqrt/cbrt/log/exp/sin/cos/tan + inverses, floor/ceil/
                              round, radians/degrees, hypot, gcd/lcm/factorial,
                              min/max/abs/pow. Constants pi/e/tau. No units, no
                              unresolved symbols. Examples: `"3847 * 29"`,
                              `"2450 * 0.15"`, `"pi * 5**2"`, `"hypot(3, 4)"`.
    """
    import server
    p = params or {}
    if action == "wikipedia":
        _require(p, "title")
        return await server.get_wikipedia_summary(p["title"], p.get("lang", "en"))
    if action == "country_info":
        _require(p, "country")
        return await server.get_country_info(p["country"])
    if action == "public_holidays":
        _require(p, "country_code")
        return await server.get_public_holidays(p["country_code"], p.get("year"))
    if action == "convert_currency":
        _require(p, "amount", "from_currency", "to_currency")
        return await server.convert_currency(
            p["amount"], p["from_currency"], p["to_currency"]
        )
    if action == "calculate":
        _require(p, "expression")
        return await server.calculate(p["expression"])
    raise ValueError(f"knowledge: unknown action {action!r}")


async def radio(params: dict | None = None) -> dict:
    """Find internet-radio streams (NOT FM/AM frequencies).

    Pass at least one filter in `params`:
      - `country`?: str — ISO-2 (`"UA"`) preferred, or full name (`"Ukraine"`)
      - `tag`?: str — genre keyword (`"jazz"`, `"news"`)
      - `language`?: str — full English name (`"russian"`, not `"ru"`)
      - `limit`?: int=5 — 1-20
    """
    import server
    p = params or {}
    return await server.list_radio_stations(
        p.get("country"), p.get("tag"), p.get("language"), p.get("limit", 5)
    )


_WEB_ACTIONS = Literal["search", "news", "hackernews", "trends"]


async def web(action: _WEB_ACTIONS, params: dict | None = None) -> dict:
    """Internet search / news / Hacker News / Google Trends — real-time info.

    Pick by user intent — all four route through this one tool:
      - `search`:     {"query": str, "limit"?: int=8}
                       general DuckDuckGo web search (docs, references,
                       static content). Use for «найди», «что такое X»,
                       non-time-sensitive queries.
      - `news`:       {"query"?: str, "topic"?: str, "lang"?: str, "limit"?: int=10}
                       recent journalism via Google News. No args →
                       top headlines for user's detected country
                       (GeoIP). `query` → news-search. `topic`
                       (e.g. "tech", "business") → category-style
                       search. Use for current events and «что
                       нового про X».
      - `hackernews`: {"category"?: "top"|"new"|"best"|"ask"|"show"|"job", "limit"?: int=15}
                       HN feed. Use when the user names HN, asks
                       about the tech community, or wants Show HN /
                       Ask HN.
      - `trends`:     {"country_code"?: str, "limit"?: int=15}
                       today's top search queries by country (GeoIP
                       default). Answers «что в трендах сегодня».

    Disambiguation in one sentence: `search` = static web, `news` =
    time-sensitive journalism, `hackernews` = tech-community feed,
    `trends` = mass-attention signal.
    """
    import server
    p = params or {}
    if action == "search":
        _require(p, "query")
        return await server.web_search(p["query"], p.get("limit", 8))
    if action == "news":
        return await server.news(
            p.get("query"), p.get("topic"), p.get("lang"), p.get("limit", 10)
        )
    if action == "hackernews":
        return await server.hackernews(p.get("category", "top"), p.get("limit", 15))
    if action == "trends":
        return await server.trends(p.get("country_code"), p.get("limit", 15))
    raise ValueError(f"web: unknown action {action!r}")


def install_fat_tools_lean(mcp) -> None:
    """Register the 5 lean fat-domain tools on the FastMCP instance.

    Called from `server._install_router()` only when mode is
    `fat_tools_lean`. The narrow `@mcp.tool`s remain registered but
    hidden by the list_tools override (same pattern as fat_tools).
    """
    mcp.tool()(weather)
    mcp.tool()(geo)
    mcp.tool()(knowledge)
    mcp.tool()(radio)
    mcp.tool()(web)


def _require(params: dict, *keys: str) -> None:
    """Raise ValueError if any required key is missing / empty.

    Error message lists all missing fields so the model can retry
    in one shot. Same shape as fat_tools.py's helper — kept separate
    because lean stays fully independent of the fatter sibling.
    """
    missing = [k for k in keys if params.get(k) in (None, "", [])]
    if missing:
        raise ValueError(f"missing required param(s) in `params`: {', '.join(missing)}")
