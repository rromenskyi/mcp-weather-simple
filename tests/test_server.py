"""Functional tests for the MCP weather server.

Network calls are mocked with `respx` — no external traffic, no
API-key concerns, fast. Each test is a focused assertion on a single
tool's contract (input handling + shape of the response).
"""

from __future__ import annotations

from datetime import date
from unittest.mock import patch

import httpx
import pytest
import respx

import server


# ── Pure helpers ─────────────────────────────────────────────────────────


def test_day_label_anchors_to_today():
    today = date(2026, 4, 20)
    assert server._day_label(date(2026, 4, 20), today) == "today"
    assert server._day_label(date(2026, 4, 21), today) == "tomorrow"
    assert server._day_label(date(2026, 4, 19), today) == "yesterday"
    assert server._day_label(date(2026, 4, 25), today) == "in 5 days"
    assert server._day_label(date(2026, 4, 15), today) == "5 days ago"


def test_feature_type_maps_known_codes_and_falls_back():
    assert server._feature_type("PPL") == "city"
    assert server._feature_type("PPLA2") == "city"
    assert server._feature_type("MT") == "mountain"
    assert server._feature_type("LK") == "lake"
    assert server._feature_type("PPLX") == "neighborhood"
    # Unknown codes fall back to the raw code (so the caller still
    # sees *something* informative).
    assert server._feature_type("AIRB") == "AIRB"
    assert server._feature_type(None) is None


# ── Geocoding ────────────────────────────────────────────────────────────


@respx.mock
async def test_geocode_country_code_filter_prefers_matching_country():
    # Two "Moscow" candidates — one in Russia, one in the US state of Idaho.
    # Without `country_code` the first wins; with `country_code="US"` the
    # filter picks the US hit.
    respx.get(server.GEOCODE_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {"name": "Moscow", "country": "Russia", "country_code": "RU",
                     "latitude": 55.75, "longitude": 37.62, "timezone": "Europe/Moscow",
                     "feature_code": "PPLC", "admin1": "Moscow"},
                    {"name": "Moscow", "country": "United States", "country_code": "US",
                     "latitude": 46.73, "longitude": -117.00, "timezone": "America/Los_Angeles",
                     "feature_code": "PPLA2", "admin1": "Idaho"},
                ]
            },
        )
    )

    hit_default = await server._geocode("Moscow")
    assert hit_default["country_code"] == "RU"

    hit_us = await server._geocode("Moscow", country_code="us")  # case-insensitive
    assert hit_us["country_code"] == "US"
    assert hit_us["admin1"] == "Idaho"


@respx.mock
async def test_geocode_raises_when_empty():
    respx.get(server.GEOCODE_URL).mock(return_value=httpx.Response(200, json={}))
    with pytest.raises(ValueError, match="City not found"):
        await server._geocode("Nowhereville", country_code="XX")


@respx.mock
async def test_geocode_exposes_feature_type_and_postcodes():
    respx.get(server.GEOCODE_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {"name": "Beverly Hills", "country": "United States",
                     "country_code": "US", "admin1": "California",
                     "latitude": 34.07, "longitude": -118.40,
                     "timezone": "America/Los_Angeles",
                     "feature_code": "PPL", "postcodes": ["90210", "90211"]},
                ]
            },
        )
    )
    hit = await server._geocode("90210")
    assert hit["feature_type"] == "city"
    assert "90210" in hit["postcodes"]


# ── search_places feature_types filter ───────────────────────────────────


@respx.mock
async def test_search_places_filters_by_feature_type():
    respx.get(server.GEOCODE_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {"name": "Bountiful", "country_code": "US", "admin1": "Utah",
                     "latitude": 40.88, "longitude": -111.88,
                     "feature_code": "PPL", "population": 43784},
                    {"name": "Bountiful Peak", "country_code": "US", "admin1": "Utah",
                     "latitude": 40.96, "longitude": -111.81,
                     "feature_code": "MT"},
                    {"name": "Bountiful Islands", "country_code": "AU",
                     "latitude": -16.68, "longitude": 139.85,
                     "feature_code": "ISLS"},
                ]
            },
        )
    )

    all_places = await server.search_places("Bountiful")
    assert all_places["ambiguous"] is True
    assert len(all_places["candidates"]) == 3

    cities_only = await server.search_places("Bountiful", feature_types=["city"])
    assert [c["name"] for c in cities_only["candidates"]] == ["Bountiful"]

    mountains_only = await server.search_places(
        "Bountiful", feature_types=["mountain"]
    )
    assert [c["name"] for c in mountains_only["candidates"]] == ["Bountiful Peak"]


# ── Weather forecast: day_label anchoring ────────────────────────────────


@respx.mock
async def test_weather_forecast_labels_today_and_tomorrow_in_city_tz():
    # Geocoder returns a city in Europe/Kyiv.
    respx.get(server.GEOCODE_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {"name": "Kyiv", "country": "Ukraine", "country_code": "UA",
                     "latitude": 50.45, "longitude": 30.52, "timezone": "Europe/Kyiv",
                     "feature_code": "PPLC", "admin1": "Kyiv City"},
                ]
            },
        )
    )
    # Forecast API returns three days: yesterday, today, tomorrow relative to
    # a fixed "today" we inject via ZoneInfo patching isn't needed — we
    # just feed a stable date triple and patch datetime.now.
    respx.get(server.FORECAST_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "daily": {
                    "time": ["2026-04-19", "2026-04-20", "2026-04-21"],
                    "weather_code": [1, 2, 3],
                    "temperature_2m_max": [15.0, 17.0, 19.0],
                    "temperature_2m_min": [5.0, 7.0, 9.0],
                    "precipitation_sum": [0.0, 0.1, 2.0],
                    "precipitation_probability_max": [10, 20, 60],
                    "wind_speed_10m_max": [12.0, 15.0, 18.0],
                },
            },
        )
    )

    class FrozenDT:
        @classmethod
        def now(cls, tz=None):
            from datetime import datetime
            return datetime(2026, 4, 20, 12, 0, tzinfo=tz)

    with patch.object(server, "datetime", FrozenDT):
        result = await server.get_weather_forecast("Kyiv", days=3)

    labels = [d["day_label"] for d in result["days"]]
    assert labels == ["yesterday", "today", "tomorrow"]
    assert result["timezone_id"] == "Europe/Kyiv"


# ── Historical: date validation ──────────────────────────────────────────


async def test_historical_weather_rejects_reversed_dates():
    with pytest.raises(ValueError, match="precedes"):
        await server.get_historical_weather(
            "Kyiv", start_date_iso="2024-01-10", end_date_iso="2024-01-01"
        )


async def test_historical_weather_rejects_spans_over_31_days():
    with pytest.raises(ValueError, match="31 days"):
        await server.get_historical_weather(
            "Kyiv", start_date_iso="2024-01-01", end_date_iso="2024-03-01"
        )


# ── GeoIP + no-arg shortcut chain ────────────────────────────────────────


@respx.mock
async def test_geoip_detect_returns_clock_fields():
    respx.get(f"{server.GEOIP_URL}/").mock(
        return_value=httpx.Response(
            200,
            json={
                "success": True,
                "ip": "203.0.113.10",
                "city": "Kyiv",
                "region": "Kyiv City",
                "country": "Ukraine",
                "country_code": "UA",
                "latitude": 50.45,
                "longitude": 30.52,
                "timezone": {"id": "Europe/Kyiv"},
            },
        )
    )

    loc = await server.detect_my_location_by_ip()
    assert loc["city"] == "Kyiv"
    assert loc["timezone_id"] == "Europe/Kyiv"
    # Every clock field populated — not just whatever ipwho returned.
    for key in ("local_time", "local_date", "weekday", "iso_datetime", "utc_offset"):
        assert key in loc and loc[key], f"{key} missing from response"


@respx.mock
async def test_shortcut_for_tomorrow_uses_geoip_then_forecast():
    respx.get(f"{server.GEOIP_URL}/").mock(
        return_value=httpx.Response(
            200,
            json={
                "success": True, "ip": "203.0.113.10", "city": "Kyiv",
                "country": "Ukraine", "country_code": "UA",
                "latitude": 50.45, "longitude": 30.52,
                "timezone": {"id": "Europe/Kyiv"},
            },
        )
    )
    respx.get(server.GEOCODE_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {"name": "Kyiv", "country": "Ukraine", "country_code": "UA",
                     "latitude": 50.45, "longitude": 30.52,
                     "timezone": "Europe/Kyiv",
                     "feature_code": "PPLC", "admin1": "Kyiv City"},
                ]
            },
        )
    )
    respx.get(server.FORECAST_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "daily": {
                    "time": ["2026-04-20", "2026-04-21"],
                    "weather_code": [2, 3],
                    "temperature_2m_max": [17.0, 19.0],
                    "temperature_2m_min": [7.0, 9.0],
                    "precipitation_sum": [0.1, 2.0],
                    "precipitation_probability_max": [20, 60],
                    "wind_speed_10m_max": [15.0, 18.0],
                },
            },
        )
    )

    class FrozenDT:
        @classmethod
        def now(cls, tz=None):
            from datetime import datetime
            return datetime(2026, 4, 20, 12, 0, tzinfo=tz)

    with patch.object(server, "datetime", FrozenDT):
        result = await server.get_weather_forecast_for_tomorrow()

    assert [d["day_label"] for d in result["days"]] == ["today", "tomorrow"]
    assert result["location_source"] == "geoip_autodetected"
    assert "accuracy_warning" in result


# ── Air quality shape ────────────────────────────────────────────────────


@respx.mock
async def test_air_quality_returns_both_aqi_scales():
    respx.get(server.GEOCODE_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {"name": "Delhi", "country": "India", "country_code": "IN",
                     "latitude": 28.61, "longitude": 77.20,
                     "timezone": "Asia/Kolkata",
                     "feature_code": "PPLC", "admin1": "Delhi"},
                ]
            },
        )
    )
    respx.get(server.AIR_QUALITY_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "current": {
                    "time": "2026-04-20T12:00",
                    "european_aqi": 95,
                    "us_aqi": 110,
                    "pm2_5": 38.0, "pm10": 70.0, "ozone": 55.0,
                    "nitrogen_dioxide": 12.0,
                    "sulphur_dioxide": 4.0,
                    "carbon_monoxide": 300.0,
                },
            },
        )
    )

    aq = await server.get_air_quality("Delhi")
    assert aq["european_aqi"] == 95
    assert aq["us_aqi"] == 110
    assert aq["pm2_5_ugm3"] == 38.0


# ── Places / knowledge tools ─────────────────────────────────────────────


@respx.mock
async def test_wikipedia_summary_shape():
    respx.get(
        server.WIKIPEDIA_SUMMARY_URL.format(lang="en", title="Kyiv")
    ).mock(
        return_value=httpx.Response(
            200,
            json={
                "title": "Kyiv",
                "description": "Capital of Ukraine",
                "extract": "Kyiv is the capital and most populous city of Ukraine.",
                "content_urls": {"desktop": {"page": "https://en.wikipedia.org/wiki/Kyiv"}},
            },
        )
    )
    r = await server.get_wikipedia_summary("Kyiv")
    assert r["title"] == "Kyiv"
    assert r["url"].endswith("/Kyiv")
    assert r["lang"] == "en"


@respx.mock
async def test_currency_conversion_multiplies_rate():
    respx.get(server.CURRENCY_URL.format(base="USD")).mock(
        return_value=httpx.Response(
            200,
            json={
                "result": "success",
                "rates": {"EUR": 0.93, "UAH": 41.5},
                "time_last_update_utc": "Sat, 19 Apr 2026 00:00:00 +0000",
            },
        )
    )
    r = await server.convert_currency(50.0, "usd", "eur")
    assert r["from"] == "USD" and r["to"] == "EUR"
    assert r["rate"] == 0.93
    assert r["converted"] == round(50.0 * 0.93, 4)


@respx.mock
async def test_currency_rejects_unknown_target():
    respx.get(server.CURRENCY_URL.format(base="USD")).mock(
        return_value=httpx.Response(200, json={"result": "success", "rates": {"EUR": 0.93}})
    )
    with pytest.raises(ValueError, match="Unknown target currency"):
        await server.convert_currency(1.0, "USD", "XYZ")


async def test_radio_stations_requires_a_filter():
    with pytest.raises(ValueError, match="filter"):
        await server.list_radio_stations()


# ── Fetch helper: timeouts and mirror fallback ───────────────────────────


@respx.mock
async def test_fetch_json_friendly_timeout_error():
    respx.get("https://example.test/slow").mock(side_effect=httpx.ReadTimeout("slow"))
    with pytest.raises(RuntimeError, match="did not respond within"):
        await server._fetch_json("https://example.test/slow", service="TestService", timeout=1.0)


@respx.mock
async def test_fetch_json_falls_back_to_second_mirror():
    respx.get("https://a.example.test/").mock(side_effect=httpx.ConnectError("down"))
    respx.get("https://b.example.test/").mock(
        return_value=httpx.Response(200, json={"ok": True})
    )
    out = await server._fetch_json(
        ["https://a.example.test/", "https://b.example.test/"],
        service="TestService",
        timeout=2.0,
    )
    assert out == {"ok": True}


@respx.mock
async def test_fetch_json_raises_after_all_mirrors_fail():
    respx.get("https://a.example.test/").mock(side_effect=httpx.ConnectError("down"))
    respx.get("https://b.example.test/").mock(
        return_value=httpx.Response(503, json={})
    )
    with pytest.raises(RuntimeError, match="tried 2 mirror"):
        await server._fetch_json(
            ["https://a.example.test/", "https://b.example.test/"],
            service="TestService",
            timeout=2.0,
        )


# ── Coordinate validation ────────────────────────────────────────────────


async def test_weather_by_coordinates_rejects_out_of_range_lat():
    with pytest.raises(ValueError, match="latitude"):
        await server.get_weather_by_coordinates(latitude=100.0, longitude=0.0)


async def test_weather_by_coordinates_rejects_out_of_range_lon():
    with pytest.raises(ValueError, match="longitude"):
        await server.get_weather_by_coordinates(latitude=0.0, longitude=-200.0)


# ── Script detection for multilingual queries ────────────────────────────


def test_detect_query_language_maps_common_scripts():
    assert server._detect_query_language("Paris") == "en"
    assert server._detect_query_language("Москва") == "ru"        # Russian Cyrillic
    assert server._detect_query_language("Київ") == "ru"          # Ukrainian Cyrillic — same script bucket
    assert server._detect_query_language("Αθήνα") == "el"         # Greek
    assert server._detect_query_language("القاهرة") == "ar"       # Arabic
    assert server._detect_query_language("ירושלים") == "he"       # Hebrew
    assert server._detect_query_language("北京") == "zh"           # CJK Han
    assert server._detect_query_language("東京") == "ja" or \
           server._detect_query_language("東京") == "zh"           # Kanji overlap; either is acceptable
    assert server._detect_query_language("東京タワー") == "ja"      # Katakana forces Japanese
    assert server._detect_query_language("서울") == "ko"           # Hangul


@respx.mock
async def test_geocode_passes_detected_language_for_cyrillic_queries():
    # When the query is Cyrillic, the helper should request
    # `language=ru` so Open-Meteo indexes Russian-native names instead
    # of falling back to the Latin-only index (which returns only
    # tiny Tajik villages named "Moskva" for the query "Москва").
    route = respx.get(server.GEOCODE_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {"name": "Москва", "country": "Россия", "country_code": "RU",
                     "latitude": 55.75, "longitude": 37.62,
                     "timezone": "Europe/Moscow", "feature_code": "PPLC",
                     "admin1": "Moscow", "population": 10381222},
                ]
            },
        )
    )
    hit = await server._geocode("Москва")
    assert hit["country_code"] == "RU"
    assert route.calls.last.request.url.params["language"] == "ru"
