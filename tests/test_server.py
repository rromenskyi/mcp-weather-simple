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
async def test_geocode_error_is_self_correcting_for_comma_separated_queries():
    # Open-Meteo treats `name` as a literal match, so a postal-address
    # string like "Bountiful, Utah, 84010" yields zero results even
    # though the first token is a real city. The error must hand the
    # model a corrective hint — single-token contract, country_code
    # suggestion, split-on-comma rewrite — instead of a generic
    # "not found" that the LLM cannot recover from.
    respx.get(server.GEOCODE_URL).mock(return_value=httpx.Response(200, json={"results": []}))
    with pytest.raises(ValueError) as excinfo:
        await server._geocode("Bountiful, Utah, 84010")
    msg = str(excinfo.value)
    assert "SINGLE token" in msg
    assert "'Bountiful'" in msg  # split-on-comma suggestion
    assert "country_code" in msg


@respx.mock
async def test_search_places_returns_hint_when_comma_query_is_empty():
    # search_places doesn't raise (returns empty candidates), but the
    # hint field surfaces the same self-correcting contract so the
    # LLM can re-call without a second user prompt.
    respx.get(server.GEOCODE_URL).mock(return_value=httpx.Response(200, json={"results": []}))
    resp = await server.search_places("Bountiful, Utah, 84010")
    assert resp["candidates"] == []
    assert "hint" in resp
    assert "SINGLE token" in resp["hint"]


@respx.mock
async def test_geocode_error_stays_generic_without_comma():
    respx.get(server.GEOCODE_URL).mock(return_value=httpx.Response(200, json={"results": []}))
    with pytest.raises(ValueError) as excinfo:
        await server._geocode("Nowhereville", country_code="XX")
    msg = str(excinfo.value)
    # No comma → no self-correcting hint, just the short form.
    assert "City not found: Nowhereville in XX" in msg
    assert "SINGLE token" not in msg


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


@respx.mock
async def test_radio_stations_iso2_country_routes_to_exact_code_endpoint():
    # A live 2026-04-20 mcphost session called
    # list_radio_stations(country="US") and got back Russian /
    # Australian stations because /stations/bycountry/ does a fuzzy
    # substring match. Routing ISO-2 tokens to
    # /stations/bycountrycodeexact/ fixes it; this test pins the
    # routing decision.
    exact = respx.get(
        f"{server.RADIO_BROWSER_MIRRORS[0]}/stations/bycountrycodeexact/US"
    ).mock(return_value=httpx.Response(200, json=[]))
    byname = respx.get(
        f"{server.RADIO_BROWSER_MIRRORS[0]}/stations/bycountry/US"
    ).mock(return_value=httpx.Response(200, json=[]))
    await server.list_radio_stations(country="US")
    assert exact.called
    assert not byname.called


@respx.mock
async def test_radio_stations_full_country_name_routes_to_byname():
    byname = respx.get(
        f"{server.RADIO_BROWSER_MIRRORS[0]}/stations/bycountry/United%20States"
    ).mock(return_value=httpx.Response(200, json=[]))
    await server.list_radio_stations(country="United States")
    assert byname.called


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


def test_detect_query_languages_returns_fallback_chains():
    # Latin — single-item list, no fallback needed.
    assert server._detect_query_languages("Paris") == ["en"]
    # Generic Cyrillic — Russian first, Ukrainian second (catches "Одеса",
    # which has only shared Cyrillic glyphs and returns empty under `ru`).
    assert server._detect_query_languages("Москва") == ["ru", "uk"]
    assert server._detect_query_languages("Одеса") == ["ru", "uk"]
    # Ukrainian-unique glyphs short-circuit to `uk` first (Київ has `ї`,
    # Львів has `і`) so we never waste an empty round trip on `ru`.
    assert server._detect_query_languages("Київ") == ["uk", "ru"]
    assert server._detect_query_languages("Львів") == ["uk", "ru"]
    assert server._detect_query_languages("Харків") == ["uk", "ru"]
    # Other Cyrillic-adjacent scripts stay deterministic.
    assert server._detect_query_languages("Αθήνα") == ["el"]
    assert server._detect_query_languages("القاهرة") == ["ar"]
    assert server._detect_query_languages("ירושלים") == ["he"]
    # Han alone is ambiguous — zh / ja / ko fallback catches
    # cross-script city names (Yokohama "横浜" resolves only under `ja`).
    assert server._detect_query_languages("北京") == ["zh", "ja", "ko"]
    assert server._detect_query_languages("東京") == ["zh", "ja", "ko"]
    assert server._detect_query_languages("横浜") == ["zh", "ja", "ko"]
    # Kana / Hangul fully specify the language.
    assert server._detect_query_languages("東京タワー") == ["ja"]
    assert server._detect_query_languages("서울") == ["ko"]


def test_detect_query_language_primary_is_first_of_list():
    # Thin alias used by code that only wants the primary pick.
    assert server._detect_query_language("Paris") == "en"
    assert server._detect_query_language("Москва") == "ru"
    assert server._detect_query_language("Київ") == "uk"
    assert server._detect_query_language("横浜") == "zh"


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


@respx.mock
async def test_geocode_falls_back_from_ru_to_uk_for_ukrainian_only_cities():
    # "Одеса" (Ukrainian spelling) contains no Ukrainian-unique glyphs,
    # so the detector orders `ru` before `uk`. Open-Meteo actually
    # returns nothing under `ru` for this query — the helper must retry
    # with `uk` and succeed there. Proves the fallback loop works for
    # mixed-script city names.
    def respond(request: httpx.Request) -> httpx.Response:
        lang = request.url.params.get("language")
        if lang == "ru":
            return httpx.Response(200, json={"results": []})
        if lang == "uk":
            return httpx.Response(
                200,
                json={
                    "results": [
                        {"name": "Одеса", "country": "Україна",
                         "country_code": "UA", "admin1": "Odesa",
                         "latitude": 46.48, "longitude": 30.72,
                         "timezone": "Europe/Kyiv",
                         "feature_code": "PPLA", "population": 1015826},
                    ]
                },
            )
        return httpx.Response(200, json={"results": []})

    respx.get(server.GEOCODE_URL).mock(side_effect=respond)
    hit = await server._geocode("Одеса")
    assert hit["country_code"] == "UA"
    assert hit["population"] == 1015826


@respx.mock
async def test_geocode_falls_back_from_zh_to_ja_for_han_only_japanese_cities():
    # Yokohama ("横浜") resolves to a nonsense Chinese hit under `zh`
    # (no population, low-quality feature), and to the real Japanese
    # city only under `ja`. Proves the Han fallback chain catches the
    # case even when `zh` returned *something* — we take the first
    # non-empty match though, so this test drives `zh` to empty to
    # keep the assertion crisp and future-proof.
    def respond(request: httpx.Request) -> httpx.Response:
        lang = request.url.params.get("language")
        if lang == "ja":
            return httpx.Response(
                200,
                json={
                    "results": [
                        {"name": "横浜", "country": "日本",
                         "country_code": "JP", "admin1": "Kanagawa",
                         "latitude": 35.44, "longitude": 139.65,
                         "timezone": "Asia/Tokyo",
                         "feature_code": "PPLA", "population": 4412},
                    ]
                },
            )
        return httpx.Response(200, json={"results": []})

    respx.get(server.GEOCODE_URL).mock(side_effect=respond)
    hit = await server._geocode("横浜")
    assert hit["country_code"] == "JP"


@respx.mock
async def test_detect_my_location_by_ip_warning_differs_for_explicit_ip():
    respx.get(server.GEOCODE_URL).mock(return_value=httpx.Response(200, json={}))
    respx.get(f"{server.GEOIP_URL}/").mock(
        return_value=httpx.Response(
            200,
            json={
                "success": True, "ip": "203.0.113.10", "city": "Auto",
                "country": "Nowhere", "country_code": "ZZ",
                "latitude": 0.0, "longitude": 0.0,
                "timezone": {"id": "UTC"},
            },
        )
    )
    respx.get(f"{server.GEOIP_URL}/198.51.100.7").mock(
        return_value=httpx.Response(
            200,
            json={
                "success": True, "ip": "198.51.100.7", "city": "Manual",
                "country": "Nowhere", "country_code": "ZZ",
                "latitude": 0.0, "longitude": 0.0,
                "timezone": {"id": "UTC"},
            },
        )
    )

    auto = await server.detect_my_location_by_ip()
    assert auto["location_source"] == "geoip_autodetected"
    assert "caller's public IP address" in auto["accuracy_warning"]

    explicit = await server.detect_my_location_by_ip(ip="198.51.100.7")
    assert explicit["location_source"] == "geoip_explicit"
    assert "198.51.100.7" in explicit["accuracy_warning"]
    assert "caller's public IP address" not in explicit["accuracy_warning"]


# ── Happy paths for the remaining tools ──────────────────────────────────


@respx.mock
async def test_hourly_forecast_returns_per_hour_rows():
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
    respx.get(server.FORECAST_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "hourly": {
                    "time": ["2026-04-20T09:00", "2026-04-20T10:00", "2026-04-20T11:00"],
                    "temperature_2m": [14.0, 15.5, 16.8],
                    "relative_humidity_2m": [55, 50, 48],
                    "precipitation": [0.0, 0.0, 0.3],
                    "precipitation_probability": [0, 10, 40],
                    "weather_code": [1, 2, 61],
                    "cloud_cover": [20, 40, 80],
                    "wind_speed_10m": [8.0, 10.0, 13.0],
                    "wind_direction_10m": [200, 210, 220],
                },
            },
        )
    )

    result = await server.get_hourly_forecast("Kyiv", hours=3)
    assert result["timezone_id"] == "Europe/Kyiv"
    assert len(result["hours"]) == 3
    assert result["hours"][0]["temperature_c"] == 14.0
    assert result["hours"][2]["precipitation_probability_pct"] == 40
    assert result["hours"][2]["conditions"] == "slight rain"


@respx.mock
async def test_sunrise_sunset_formats_daylight_duration_as_hhmm():
    respx.get(server.GEOCODE_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {"name": "Reykjavík", "country": "Iceland", "country_code": "IS",
                     "latitude": 64.14, "longitude": -21.90,
                     "timezone": "Atlantic/Reykjavik",
                     "feature_code": "PPLC", "admin1": "Höfuðborgarsvæðið"},
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
                    "sunrise": ["2026-04-20T06:04", "2026-04-21T06:00"],
                    "sunset": ["2026-04-20T20:48", "2026-04-21T20:52"],
                    # 14 h 44 min in seconds — proves the hh:mm formatting.
                    "daylight_duration": [53040, 53520],
                    "sunshine_duration": [32400, 0],
                },
            },
        )
    )

    r = await server.get_sunrise_sunset("Reykjavík", days=2)
    assert r["timezone_id"] == "Atlantic/Reykjavik"
    assert r["days"][0]["daylight_duration_hhmm"] == "14:44"
    assert r["days"][1]["sunshine_duration_hhmm"] == "00:00"


@respx.mock
async def test_country_info_flattens_rest_countries_shape():
    respx.get(server.RESTCOUNTRIES_ALPHA_URL.format(code="UA")).mock(
        return_value=httpx.Response(
            200,
            json=[{
                "name": {"common": "Ukraine", "official": "Ukraine"},
                "cca2": "UA",
                "capital": ["Kyiv"],
                "region": "Europe", "subregion": "Eastern Europe",
                "population": 44134693, "area": 603500,
                "currencies": {"UAH": {"name": "Ukrainian hryvnia", "symbol": "₴"}},
                "languages": {"ukr": "Ukrainian"},
                "idd": {"root": "+3", "suffixes": ["80"]},
                "borders": ["BLR", "HUN", "MDA", "POL", "ROU", "RUS", "SVK"],
                "timezones": ["UTC+02:00"],
                "flag": "🇺🇦",
            }],
        )
    )

    r = await server.get_country_info("UA")
    assert r["name"] == "Ukraine"
    assert r["capital"] == "Kyiv"
    assert r["calling_code"] == "+380"
    assert r["currencies"] == [{"code": "UAH", "name": "Ukrainian hryvnia", "symbol": "₴"}]
    assert r["languages"] == ["Ukrainian"]
    assert r["flag_emoji"] == "🇺🇦"


@respx.mock
async def test_public_holidays_maps_nager_payload():
    respx.get(server.HOLIDAYS_URL.format(year=2026, country_code="UA")).mock(
        return_value=httpx.Response(
            200,
            json=[
                {"date": "2026-01-01", "localName": "Новий рік",
                 "name": "New Year's Day", "fixed": True, "global": True,
                 "types": ["Public"]},
                {"date": "2026-08-24", "localName": "День Незалежності",
                 "name": "Independence Day", "fixed": True, "global": True,
                 "types": ["Public"]},
            ],
        )
    )

    r = await server.get_public_holidays("ua", year=2026)
    assert r["country_code"] == "UA"
    assert r["year"] == 2026
    assert [h["date"] for h in r["holidays"]] == ["2026-01-01", "2026-08-24"]
    assert r["holidays"][1]["local_name"] == "День Незалежності"


@respx.mock
async def test_fetch_json_4xx_is_labelled_as_caller_error_not_service_outage():
    # 404 from Wikipedia (title doesn't exist) should tell the model
    # to fix its argument, not "service may be having issues". 500 is
    # the opposite — upstream problem, retry/report to user.
    respx.get("https://example.test/missing").mock(
        return_value=httpx.Response(404, json={"detail": "not found"})
    )
    respx.get("https://example.test/broken").mock(
        return_value=httpx.Response(502, json={})
    )
    with pytest.raises(RuntimeError, match="rejected the request with HTTP 404"):
        await server._fetch_json("https://example.test/missing", service="TestSvc", timeout=1.0)
    with pytest.raises(RuntimeError, match="service may be having issues"):
        await server._fetch_json("https://example.test/broken", service="TestSvc", timeout=1.0)


@respx.mock
async def test_geocode_uses_count_100_when_country_code_is_set():
    # London, CA has lower population than London, UK → sits outside
    # the default count=10 Open-Meteo page. country_code="CA" must
    # bump the page size so the filter can still find it.
    route = respx.get(server.GEOCODE_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {"name": "London", "country_code": "CA", "admin1": "Ontario",
                     "latitude": 42.98, "longitude": -81.25,
                     "timezone": "America/Toronto",
                     "feature_code": "PPLA2", "population": 383822},
                ]
            },
        )
    )
    hit = await server._geocode("London", country_code="CA")
    assert hit["country_code"] == "CA"
    assert route.calls.last.request.url.params["count"] == "100"


@respx.mock
async def test_geocode_uses_count_10_when_no_country_code_filter():
    route = respx.get(server.GEOCODE_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {"name": "London", "country_code": "GB", "admin1": "England",
                     "latitude": 51.51, "longitude": -0.13,
                     "timezone": "Europe/London",
                     "feature_code": "PPLC", "population": 8900000},
                ]
            },
        )
    )
    await server._geocode("London")
    assert route.calls.last.request.url.params["count"] == "10"


@respx.mock
async def test_list_radio_stations_sorts_by_clickcount_and_trims_to_limit():
    # radio-browser returns 4 stations; we request limit=2 and expect
    # the two with the highest `clickcount` in that order.
    respx.get(f"{server.RADIO_BROWSER_MIRRORS[0]}/stations/bycountry/Ukraine").mock(
        return_value=httpx.Response(
            200,
            json=[
                {"name": "Radio A", "country": "Ukraine", "language": "ukrainian",
                 "tags": "news,talk", "url_resolved": "https://a", "homepage": "https://a.home",
                 "bitrate": 128, "codec": "MP3", "clickcount": 5000},
                {"name": "Radio B", "country": "Ukraine", "language": "ukrainian",
                 "tags": "music", "url_resolved": "https://b", "homepage": "https://b.home",
                 "bitrate": 192, "codec": "MP3", "clickcount": 20000},
                {"name": "Radio C", "country": "Ukraine", "language": "ukrainian",
                 "tags": "jazz", "url_resolved": "https://c", "homepage": "https://c.home",
                 "bitrate": 128, "codec": "AAC", "clickcount": 12000},
                {"name": "Radio D", "country": "Ukraine", "language": "ukrainian",
                 "tags": "rock", "url_resolved": "https://d", "homepage": "https://d.home",
                 "bitrate": 128, "codec": "MP3", "clickcount": 800},
            ],
        )
    )

    r = await server.list_radio_stations(country="Ukraine", limit=2)
    assert r["count"] == 2
    assert [s["name"] for s in r["stations"]] == ["Radio B", "Radio C"]
    # User-Agent header is a radio-browser ToS requirement — make sure
    # we always send it with the repo-identifying string.
    last_req = respx.calls.last.request
    assert "mcp-weather-simple" in last_req.headers.get("user-agent", "")
