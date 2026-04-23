"""Narrow tool → (fat_tool, action) mapping shared between the
router-mode dispatchers and the eval scorer.

Lives in its own module with zero other imports so the eval harness
can load it without dragging the whole `server` module-init chain
(which would trigger `_install_router` recursion when
`MCP_ROUTER_MODE=fat_tools` is set).
"""

from __future__ import annotations


# Rename any action here = rename in `fat_tools.py`'s matching
# dispatcher (and vice versa). A unit test in tests/test_server.py
# enforces 1:1 coverage against the narrow tool registry.
NARROW_TO_FAT: dict[str, tuple[str, str | None]] = {
    # weather
    "get_weather_outside_right_now":     ("weather", "current_here"),
    "get_weather_forecast_for_today":    ("weather", "today_here"),
    "get_weather_forecast_for_tomorrow": ("weather", "tomorrow_here"),
    "get_current_weather_in_city":       ("weather", "current_in_city"),
    "get_weather_forecast":              ("weather", "forecast_days"),
    "get_hourly_forecast":               ("weather", "hourly"),
    "get_sunrise_sunset":                ("weather", "sunrise_sunset"),
    "get_air_quality":                   ("weather", "air_quality"),
    "get_weather_by_coordinates":        ("weather", "by_coordinates"),
    "get_historical_weather":            ("weather", "historical"),
    # geo
    "find_place_coordinates":            ("geo", "find_coordinates"),
    "search_places":                     ("geo", "search_places"),
    "resolve_address":                   ("geo", "resolve_address"),
    "detect_my_location_by_ip":          ("geo", "detect_my_location"),
    "lookup_ip_geolocation":             ("geo", "lookup_ip"),
    "get_current_time_where_i_am":       ("geo", "time_here"),
    "get_current_time_in_city":          ("geo", "time_in_city"),
    "get_current_date":                  ("geo", "date_in_timezone"),
    # knowledge
    "get_wikipedia_summary":             ("knowledge", "wikipedia"),
    "get_country_info":                  ("knowledge", "country_info"),
    "get_public_holidays":               ("knowledge", "public_holidays"),
    "convert_currency":                  ("knowledge", "convert_currency"),
    # radio (single entry, no action discriminator)
    "list_radio_stations":               ("radio", None),
    # knowledge — arithmetic calculator (added 2026-04-22 for small-LLM
    # math reliability; see `server._safe_eval` for the whitelist).
    "calculate":                         ("knowledge", "calculate"),
    # web — search / news / HN / trends (added 2026-04-22). All four
    # providers are no-auth; see the narrow tools' docstrings in
    # server.py for the disambiguation rules (search = static refs,
    # news = time-sensitive journalism, hackernews = tech community,
    # trends = mass-attention signal).
    "web_search":                        ("web", "search"),
    "news":                              ("web", "news"),
    "hackernews":                        ("web", "hackernews"),
    "trends":                            ("web", "trends"),
}
