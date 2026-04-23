"""Functional tests for the MCP weather server.

Network calls are mocked with `respx` — no external traffic, no
API-key concerns, fast. Each test is a focused assertion on a single
tool's contract (input handling + shape of the response).
"""

from __future__ import annotations

from datetime import date
from importlib import reload
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import respx

import server


# ── Shared fixtures ──────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _reset_loop_detector():
    # The #19 duplicate-call detector holds a process-wide deque of
    # recent fingerprints. Without a reset between tests, a tool call
    # in test A would short-circuit the same call in test B. Wiping
    # before AND after is paranoid but cheap — keeps the invariant
    # regardless of test ordering.
    server._reset_recent_calls()
    yield
    server._reset_recent_calls()


# ── Loop breaker (#19) ───────────────────────────────────────────────────


def test_call_fingerprint_is_stable_across_kwarg_order():
    # The 2026-04-20 mcphost loop flipped kwarg order on turn 9.
    # Fingerprints that differ on {country: X, language: Y} vs
    # {language: Y, country: X} would defeat the detector — sort keys.
    a = server._call_fingerprint(
        "list_radio_stations",
        {"country": "The Russian Federation", "language": "russian", "limit": 5},
    )
    b = server._call_fingerprint(
        "list_radio_stations",
        {"language": "russian", "limit": 5, "country": "The Russian Federation"},
    )
    assert a == b


def test_call_fingerprint_differs_when_arguments_differ():
    a = server._call_fingerprint("get_current_weather_in_city", {"city": "Kyiv"})
    b = server._call_fingerprint("get_current_weather_in_city", {"city": "Paris"})
    assert a != b


def test_detect_and_record_call_fires_on_second_identical_fingerprint():
    fp = "deadbeef" * 2
    assert server._detect_and_record_call(fp) is False  # first sighting
    assert server._detect_and_record_call(fp) is True   # duplicate


def test_detect_and_record_call_expires_entries_past_the_window(monkeypatch):
    # Simulate time advancing past the window — the prior fingerprint
    # should be pruned and a re-call is NOT a duplicate. Models genuinely
    # re-querying five minutes later ("what's the weather in Kyiv now?")
    # must not get stuck on a stale short-circuit.
    clock = {"t": 1000.0}

    def fake_monotonic():
        return clock["t"]

    monkeypatch.setattr(server.time, "monotonic", fake_monotonic)

    fp = "cafebabe" * 2
    assert server._detect_and_record_call(fp) is False
    clock["t"] += server._LOOP_WINDOW_SECONDS + 1  # past the window
    assert server._detect_and_record_call(fp) is False


@respx.mock
async def test_tool_second_identical_call_returns_duplicate_envelope():
    # End-to-end: call get_current_weather_in_city twice back-to-back.
    # First call goes through to Open-Meteo; second short-circuits with
    # relay_to_user=false, guidance naming the tool. The forecast API
    # is mocked once and would 500 on a second hit — test relies on
    # the second call never reaching the upstream.
    respx.get(server.GEOCODE_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {"name": "Kyiv", "country": "Ukraine", "country_code": "UA",
                     "latitude": 50.45, "longitude": 30.52, "timezone": "Europe/Kyiv",
                     "feature_code": "PPLC", "admin1": "Kyiv City", "population": 3_000_000},
                ]
            },
        )
    )
    respx.get(server.FORECAST_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "current": {
                    "time": "2026-04-21T10:00",
                    "temperature_2m": 12.3,
                    "relative_humidity_2m": 60,
                    "apparent_temperature": 11.0,
                    "precipitation": 0.0,
                    "weather_code": 2,
                    "wind_speed_10m": 10.0,
                    "wind_direction_10m": 180,
                },
            },
        )
    )

    first = await server.get_current_weather_in_city("Kyiv")
    assert first["relay_to_user"] is True
    assert "location" in first

    second = await server.get_current_weather_in_city("Kyiv")
    assert second["relay_to_user"] is False
    assert second["tool_name"] == "get_current_weather_in_city"
    assert second["duplicate_of_recent_call"] is True
    assert "Duplicate" in second["guidance"]


@respx.mock
async def test_loop_guard_ignores_kwarg_order_end_to_end():
    # Two calls that differ only in keyword order must collide on the
    # SECOND one. (find_place_coordinates takes city + country_code; we
    # pass them in opposite orders via **kwargs unpacking.)
    respx.get(server.GEOCODE_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {"name": "Kyiv", "country": "Ukraine", "country_code": "UA",
                     "latitude": 50.45, "longitude": 30.52, "timezone": "Europe/Kyiv",
                     "feature_code": "PPLC", "admin1": "Kyiv City", "population": 3_000_000},
                ]
            },
        )
    )

    first = await server.find_place_coordinates(**{"city": "Kyiv", "country_code": "UA"})
    assert first["relay_to_user"] is True

    # Reverse the kwarg order on the caller side. Python dicts preserve
    # insertion order, so sig.bind + sort_keys json is what actually
    # makes these collide on the server side.
    second = await server.find_place_coordinates(**{"country_code": "UA", "city": "Kyiv"})
    assert second["relay_to_user"] is False
    assert second["duplicate_of_recent_call"] is True


@respx.mock
async def test_loop_guard_allows_different_arguments_to_same_tool():
    # Swapping city should NOT short-circuit.
    respx.get(server.GEOCODE_URL).mock(
        side_effect=lambda req: httpx.Response(
            200,
            json={
                "results": [
                    {
                        "name": req.url.params["name"],
                        "country": "X", "country_code": "XX",
                        "latitude": 0.0, "longitude": 0.0,
                        "timezone": "UTC",
                        "feature_code": "PPLC", "population": 1_000_000,
                    }
                ]
            },
        )
    )
    a = await server.find_place_coordinates("Kyiv")
    b = await server.find_place_coordinates("Paris")
    assert a["relay_to_user"] is True
    assert b["relay_to_user"] is True


async def test_loop_guard_isolates_recent_calls_per_session():
    """Regression guard for the shared-HTTP-deployment bug: the
    duplicate-call deque used to be a single process global. Two
    chat sessions on the same pod asking the same question within
    120s would see the second caller's request short-circuit as
    `duplicate_of_recent_call`. Now the detector is keyed by the
    MCP session id via the lowlevel `request_ctx`, and direct-Python
    callers with no request context share a single fallback key.
    This test simulates two sessions by patching the session-key
    resolver between calls and asserts neither falsely short-circuits.
    """
    server._reset_recent_calls()
    # Session A fires a call — records it under its own key.
    with patch.object(server, "_current_session_key", return_value="sess-A"):
        fp = server._call_fingerprint("list_radio_stations", {"country": "UA"})
        assert server._detect_and_record_call(fp) is False  # fresh
        assert server._detect_and_record_call(fp) is True   # same session, same fp = dup

    # Session B, identical call within window, must NOT see A's entry.
    with patch.object(server, "_current_session_key", return_value="sess-B"):
        assert server._detect_and_record_call(fp) is False, (
            "session B must not inherit session A's call history"
        )


@respx.mock
async def test_shortcut_does_not_leak_inner_call_into_loop_detector():
    # Regression guard for #3 / #19 interaction. When
    # `get_weather_outside_right_now` was calling
    # `get_current_weather_in_city(city, cc)` as an @mcp.tool-decorated
    # function, BOTH calls went through @_loop_guarded. A follow-up
    # direct call to `get_current_weather_in_city("Kyiv", "UA")` with
    # the SAME resolved city would then false-positive as a duplicate.
    # After decoupling into `_get_current_weather_in_city_impl`, the
    # inner call bypasses the guard and a direct call afterwards runs
    # cleanly.
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
                     "latitude": 50.45, "longitude": 30.52, "timezone": "Europe/Kyiv",
                     "feature_code": "PPLC", "admin1": "Kyiv City",
                     "population": 3_000_000},
                ]
            },
        )
    )
    respx.get(server.FORECAST_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "current": {
                    "time": "2026-04-21T10:00",
                    "temperature_2m": 12.3, "relative_humidity_2m": 60,
                    "apparent_temperature": 11.0, "precipitation": 0.0,
                    "weather_code": 2, "wind_speed_10m": 10.0,
                    "wind_direction_10m": 180,
                },
            },
        )
    )

    # Step 1: shortcut call. Internally resolves to Kyiv via GeoIP and
    # calls the current-weather IMPL (not the @mcp.tool).
    shortcut = await server.get_weather_outside_right_now()
    assert shortcut["relay_to_user"] is True
    assert shortcut["location_source"] == "geoip_autodetected"

    # Step 2: direct call to get_current_weather_in_city with the same
    # city the shortcut resolved to. If the inner call from step 1
    # had gone through the @_loop_guarded wrapper, this second call
    # would short-circuit as a duplicate. After #3 it runs cleanly.
    direct = await server.get_current_weather_in_city("Kyiv", country_code="UA")
    assert direct.get("duplicate_of_recent_call") is None
    assert direct["relay_to_user"] is True
    assert direct["location"] == "Kyiv, Ukraine"


async def test_docstring_mode_terse_by_default():
    # Source-level docstrings are the terse versions (trimmed on
    # 2026-04-21 for CPU/mcphost prefill budget). Pin the default
    # so a mode flip elsewhere doesn't silently change wire content.
    assert server._DOCSTRING_MODE == "terse"
    tools = await server.mcp.list_tools()
    by_name = {t.name: t for t in tools}
    # Specific signal from the terse variant of resolve_address —
    # "Parse a" is in terse, "Normalise a" is in the verbose dict.
    assert by_name["resolve_address"].description.startswith("Parse a")
    # Verbose variant of find_place_coordinates has the phrase
    # "Argument contract"; terse doesn't.
    assert "Argument contract" not in by_name["find_place_coordinates"].description


async def test_docstring_mode_verbose_overrides_the_trimmed_defaults(monkeypatch):
    # Flipping the env var and re-applying should restore the pre-trim
    # descriptions on exactly the tools in `_DOCSTRINGS_VERBOSE`.
    monkeypatch.setattr(server, "_DOCSTRING_MODE", "verbose")
    original_terse = {
        name: server.mcp._tool_manager._tools[name].description
        for name in server._DOCSTRINGS_VERBOSE
    }
    server._apply_docstring_experiment()
    try:
        tools = await server.mcp.list_tools()
        by_name = {t.name: t for t in tools}
        for name, verbose_text in server._DOCSTRINGS_VERBOSE.items():
            assert by_name[name].description == verbose_text, (
                f"{name}: verbose override did not propagate to wire"
            )
        # Unrelated tools keep their source descriptions.
        assert "detect_my_location_by_ip" not in server._DOCSTRINGS_VERBOSE
    finally:
        # Restore terse on every covered tool so later tests in the
        # session see the default (autouse fixtures wipe state, but
        # description is not one of those).
        for name, terse_text in original_terse.items():
            server.mcp._tool_manager._tools[name].description = terse_text


async def test_output_schema_experiment_off_by_default():
    # Baseline: no tool declares an outputSchema — matches our
    # pre-experiment behaviour and the MCP ecosystem's plain-text
    # position (discussion #1121).
    assert server._OUTPUT_SCHEMA_MODE == "off"
    tools = await server.mcp.list_tools()
    assert tools, "expected registered tools"
    for t in tools:
        assert t.outputSchema is None, f"{t.name} unexpectedly has an outputSchema"


async def test_output_schema_experiment_on_attaches_envelope_schema(monkeypatch):
    # When the env var is flipped on and the patch function is
    # re-applied, every tool's wire-level outputSchema is the shared
    # envelope contract. `relay_to_user` + `guidance` are declared
    # required; other body fields pass through via additionalProperties.
    monkeypatch.setattr(server, "_OUTPUT_SCHEMA_MODE", "on")
    server._apply_output_schema_experiment()
    try:
        tools = await server.mcp.list_tools()
        assert tools, "expected registered tools"
        for t in tools:
            schema = t.outputSchema
            assert schema is not None, f"{t.name} missing outputSchema"
            assert set(schema["required"]) == {"relay_to_user", "guidance"}
            assert schema["additionalProperties"] is True
            assert schema["properties"]["relay_to_user"]["type"] == "boolean"
            assert schema["properties"]["guidance"]["type"] == "string"
    finally:
        # Undo for the rest of the suite — the patch is process-wide.
        for tool in server.mcp._tool_manager._tools.values():
            tool.output_schema = None


def test_instructions_are_wired_into_fastmcp():
    # FastMCP surfaces `instructions` verbatim in InitializeResult.
    # The preamble text is part of the contract — pin the three rules.
    assert "at most 3 tools" in server._INSTRUCTIONS
    assert "Never repeat an identical tool call" in server._INSTRUCTIONS
    assert "relay_to_user: false" in server._INSTRUCTIONS
    # And it's actually attached to the FastMCP instance.
    assert server.mcp.instructions == server._INSTRUCTIONS


# ── Envelope contract (#18) ──────────────────────────────────────────────


def test_respond_adds_default_envelope_fields():
    # Default path = `relay_to_user=True` + direct-relay guidance; body
    # is merged underneath so callers can't accidentally overwrite the
    # envelope.
    env = server._respond({"foo": 1, "bar": "x"})
    assert env["foo"] == 1
    assert env["bar"] == "x"
    assert env["relay_to_user"] is True
    assert env["guidance"] == server._GUIDANCE_DIRECT


def test_respond_overrides_envelope_even_when_body_sets_the_same_keys():
    # Tool bodies that accidentally include `relay_to_user` or
    # `guidance` MUST NOT be able to undermine the contract.
    env = server._respond(
        {"relay_to_user": False, "guidance": "attacker-controlled"},
        relay_to_user=True,
        guidance="trusted",
    )
    assert env["relay_to_user"] is True
    assert env["guidance"] == "trusted"


@respx.mock
async def test_search_places_guidance_reflects_candidate_count():
    # Single hit → direct relay. Two+ hits → tell the model to either
    # list or clarify. Zero hits → error-ish guidance.
    respx.get(server.GEOCODE_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {"name": "Springfield", "country_code": "US", "admin1": "IL",
                     "latitude": 39.78, "longitude": -89.65, "feature_code": "PPLA"},
                    {"name": "Springfield", "country_code": "US", "admin1": "MA",
                     "latitude": 42.10, "longitude": -72.59, "feature_code": "PPL"},
                ]
            },
        )
    )
    multi = await server.search_places("Springfield")
    assert multi["relay_to_user"] is True  # list-all-Springfields is a legitimate intent
    assert "candidates" in multi["guidance"] or "candidate" in multi["guidance"]


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
    # Two "Moscow" candidates — one in Russia (12.5 M), one in Idaho
    # (25 k). Without `country_code` the 500× population gap means
    # Moscow-RU is the obvious intent → silent top-1. With
    # `country_code="US"` the filter narrows to the Idaho hit.
    # (#17's ambiguity detector intentionally does NOT fire here: users
    # asking for "weather in Moscow" 99% mean the Russian capital, and
    # the safety net is the `country_code` knob for the other 1%.)
    respx.get(server.GEOCODE_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {"name": "Moscow", "country": "Russia", "country_code": "RU",
                     "latitude": 55.75, "longitude": 37.62, "timezone": "Europe/Moscow",
                     "feature_code": "PPLC", "admin1": "Moscow", "population": 12_500_000},
                    {"name": "Moscow", "country": "United States", "country_code": "US",
                     "latitude": 46.73, "longitude": -117.00, "timezone": "America/Los_Angeles",
                     "feature_code": "PPLA2", "admin1": "Idaho", "population": 25_000},
                ]
            },
        )
    )

    hit_default = await server._geocode("Moscow")
    assert hit_default.get("_ambiguous") is None  # population-dominated, no short-circuit
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
async def test_geocode_country_code_filter_hard_rejects_wrong_country():
    """Regression guard for the silent-fallback bug: when
    `country_code` is set but NO hit in the result set matches it,
    the geocoder used to silently revert to the unfiltered top-1
    (Moscow, Russia) instead of raising "not found in FR". Now it
    honours the pin and surfaces the miss so the model can ask the
    user to clarify rather than lying about a French Moscow."""
    respx.get(server.GEOCODE_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {"name": "Moscow", "country": "Russia", "country_code": "RU",
                     "latitude": 55.75, "longitude": 37.62, "timezone": "Europe/Moscow",
                     "feature_code": "PPLC", "admin1": "Moscow", "population": 12_500_000},
                ]
            },
        )
    )
    # `country_code="FR"` mismatches — no silent top-1 fallback.
    with pytest.raises(ValueError, match="City not found"):
        await server._geocode("Moscow", country_code="FR")


# ── Ambiguity detection (#17) ────────────────────────────────────────────


@respx.mock
async def test_geocode_intra_country_ambiguity_fires_on_comparable_populations():
    # Springfield, IL (~114k) and Springfield, MA (~155k) have
    # populations within 10× of each other — top-1 is a toss-up.
    # country_code="US" is passed but the detector still fires because
    # intra-country homonyms are exactly the "5 Bountifuls" case.
    respx.get(server.GEOCODE_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {"name": "Springfield", "country_code": "US", "admin1": "Massachusetts",
                     "latitude": 42.10, "longitude": -72.59, "feature_code": "PPL",
                     "population": 155_000},
                    {"name": "Springfield", "country_code": "US", "admin1": "Illinois",
                     "latitude": 39.78, "longitude": -89.65, "feature_code": "PPLA",
                     "population": 114_000},
                ]
            },
        )
    )
    result = await server._geocode("Springfield", country_code="US")
    assert result.get("_ambiguous") is True
    assert "comparable populations" in result["_ambiguity_reason"]


@respx.mock
async def test_geocode_does_not_fire_ambiguity_when_top_dominates_by_10x():
    # Kyiv (3 M) vs a random Tajik hamlet "Kyiv" (1.3 k) → 2300× gap.
    # Rule 2's 10× threshold comfortably ignores this noise and the
    # top hit wins silently.
    respx.get(server.GEOCODE_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {"name": "Kyiv", "country_code": "UA", "admin1": "Kyiv City",
                     "country": "Ukraine", "latitude": 50.45, "longitude": 30.52,
                     "timezone": "Europe/Kyiv",
                     "feature_code": "PPLC", "population": 3_000_000},
                    {"name": "Kyiv", "country_code": "TJ", "admin1": "Sughd",
                     "country": "Tajikistan", "latitude": 40.0, "longitude": 69.0,
                     "feature_code": "PPL", "population": 1_300},
                ]
            },
        )
    )
    hit = await server._geocode("Kyiv")
    # Single resolved hit, no ambiguity sentinel.
    assert hit.get("_ambiguous") is None
    assert hit["country_code"] == "UA"


@respx.mock
async def test_geocode_postal_code_spanning_countries_fires_rule_3():
    # 10001 resolves to both New York-US and Troyes-FR.
    respx.get(server.GEOCODE_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {"name": "New York", "country_code": "US", "admin1": "NY",
                     "latitude": 40.7, "longitude": -74.0, "feature_code": "PPL",
                     "population": 8_300_000, "postcodes": ["10001"]},
                    {"name": "Troyes", "country_code": "FR", "admin1": "Grand Est",
                     "latitude": 48.3, "longitude": 4.1, "feature_code": "PPL",
                     "population": 60_000, "postcodes": ["10001"]},
                ]
            },
        )
    )
    result = await server._geocode("10001")
    # Population ratio 8_300_000 / 60_000 = 138× so rule 2 does NOT fire;
    # but rule 1 (cross-country, no country_code) does — either reason
    # is acceptable so the test asserts only on the sentinel.
    assert result.get("_ambiguous") is True


# ── MCP elicitation (spec 2025-11-25) ────────────────────────────────────


def _fake_context_with_elicit(elicit_result) -> MagicMock:
    """Build a Context stand-in whose `.elicit` awaitable returns the
    given ElicitationResult. `_request_context` is set truthy so our
    server's "am I inside a request?" probe lets elicitation proceed.
    """
    ctx = MagicMock()
    ctx._request_context = MagicMock()  # truthy → we're "in a request"
    ctx.elicit = AsyncMock(return_value=elicit_result)
    return ctx


@respx.mock
async def test_elicitation_accept_resolves_to_chosen_candidate():
    # When the client supports elicitation AND the user picks a
    # candidate, _resolve_place returns that candidate as a regular
    # hit (no envelope) — the weather tool then proceeds normally
    # against the chosen place, as if the geocoder had returned
    # top-1 all along.
    respx.get(server.GEOCODE_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {"name": "Springfield", "country_code": "US", "admin1": "Illinois",
                     "country": "United States",
                     "latitude": 39.78, "longitude": -89.65, "feature_code": "PPLA",
                     "timezone": "America/Chicago", "population": 114_000},
                    {"name": "Springfield", "country_code": "US", "admin1": "Massachusetts",
                     "country": "United States",
                     "latitude": 42.10, "longitude": -72.59, "feature_code": "PPL",
                     "timezone": "America/New_York", "population": 155_000},
                ]
            },
        )
    )
    # User picks the Massachusetts variant. Label format matches
    # `_candidate_label`: "<name>, <admin1>, <country_code>".
    fake_result = SimpleNamespace(
        action="accept",
        data=SimpleNamespace(choice="Springfield, Massachusetts, US"),
    )
    with patch.object(server.mcp, "get_context", return_value=_fake_context_with_elicit(fake_result)):
        hit, clarify = await server._resolve_place("Springfield", country_code="US")
    assert clarify is None
    assert hit is not None
    assert hit["admin1"] == "Massachusetts"
    assert hit["latitude"] == 42.10


@respx.mock
async def test_elicitation_decline_falls_back_to_envelope():
    # When the user declines / cancels (or returns no data), we fall
    # through to the `relay_to_user=False` envelope so the LLM still
    # gets a chance to ask the user in its own turn. Same behaviour
    # as when the client doesn't support elicitation at all — the
    # envelope is the forward-compatible safety net.
    respx.get(server.GEOCODE_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {"name": "Springfield", "country_code": "US", "admin1": "Illinois",
                     "latitude": 39.78, "longitude": -89.65, "feature_code": "PPLA",
                     "population": 114_000},
                    {"name": "Springfield", "country_code": "US", "admin1": "Massachusetts",
                     "latitude": 42.10, "longitude": -72.59, "feature_code": "PPL",
                     "population": 155_000},
                ]
            },
        )
    )
    declined = SimpleNamespace(action="decline", data=None)
    with patch.object(server.mcp, "get_context", return_value=_fake_context_with_elicit(declined)):
        hit, clarify = await server._resolve_place("Springfield", country_code="US")
    assert hit is None
    assert clarify is not None
    assert clarify["relay_to_user"] is False
    assert "Springfield" in clarify["guidance"]


@respx.mock
async def test_elicitation_raised_exception_falls_back_to_envelope():
    # If the client did NOT advertise elicitation capability,
    # FastMCP raises internally when the server tries to elicit.
    # Verify we swallow it and route to the envelope — otherwise
    # OWUI / mcphost (which don't yet speak elicitation) would see
    # a crashing tool call instead of a disambiguation request.
    respx.get(server.GEOCODE_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {"name": "Springfield", "country_code": "US", "admin1": "Illinois",
                     "latitude": 39.78, "longitude": -89.65, "feature_code": "PPLA",
                     "population": 114_000},
                    {"name": "Springfield", "country_code": "US", "admin1": "Massachusetts",
                     "latitude": 42.10, "longitude": -72.59, "feature_code": "PPL",
                     "population": 155_000},
                ]
            },
        )
    )
    ctx = MagicMock()
    ctx._request_context = MagicMock()
    ctx.elicit = AsyncMock(side_effect=RuntimeError("client does not support elicitation"))
    with patch.object(server.mcp, "get_context", return_value=ctx):
        hit, clarify = await server._resolve_place("Springfield", country_code="US")
    assert hit is None
    assert clarify is not None
    assert clarify["relay_to_user"] is False


@respx.mock
async def test_weather_tool_short_circuits_ambiguity_with_relay_to_user_false():
    # End-to-end: a @mcp.tool that uses _resolve_place must return
    # relay_to_user=False + guidance naming candidates WITHOUT hitting
    # the forecast API — short-circuit happens inside _geocode.
    respx.get(server.GEOCODE_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {"name": "Springfield", "country_code": "US", "admin1": "Illinois",
                     "latitude": 39.78, "longitude": -89.65, "feature_code": "PPLA",
                     "population": 114_000},
                    {"name": "Springfield", "country_code": "US", "admin1": "Massachusetts",
                     "latitude": 42.10, "longitude": -72.59, "feature_code": "PPL",
                     "population": 155_000},
                ]
            },
        )
    )
    # FORECAST_URL deliberately NOT mocked — if short-circuit leaks, test
    # will blow up with an unexpected outbound request.
    result = await server.get_current_weather_in_city("Springfield", country_code="US")
    assert result["relay_to_user"] is False
    assert "Springfield" in result["guidance"]
    assert "ask" in result["guidance"].lower() or "pick" in result["guidance"].lower()
    assert len(result["candidates"]) == 2


@respx.mock
async def test_geoip_shortcut_propagates_inner_ambiguity_without_geoip_override():
    # get_weather_outside_right_now calls get_current_weather_in_city
    # internally. If the inner call returns relay_to_user=False (#17),
    # the shortcut must pass it through verbatim instead of re-wrapping
    # with the GeoIP guidance and silently losing the ambiguity flag.
    respx.get(f"{server.GEOIP_URL}/").mock(
        return_value=httpx.Response(
            200,
            json={
                "success": True, "ip": "203.0.113.10", "city": "Springfield",
                "country": "United States", "country_code": "US",
                "latitude": 0.0, "longitude": 0.0,
                "timezone": {"id": "America/Chicago"},
            },
        )
    )
    respx.get(server.GEOCODE_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {"name": "Springfield", "country_code": "US", "admin1": "Illinois",
                     "latitude": 39.78, "longitude": -89.65, "feature_code": "PPLA",
                     "population": 114_000},
                    {"name": "Springfield", "country_code": "US", "admin1": "Massachusetts",
                     "latitude": 42.10, "longitude": -72.59, "feature_code": "PPL",
                     "population": 155_000},
                ]
            },
        )
    )
    result = await server.get_weather_outside_right_now()
    assert result["relay_to_user"] is False
    # Guidance is the ambiguity one, NOT the GeoIP caveat.
    assert "Springfield" in result["guidance"]
    assert "GeoIP" not in result["guidance"] and "caveat" not in result["guidance"].lower()


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


# ── resolve_address (#16) ────────────────────────────────────────────────


_NOMINATIM_HIT_BOUNTIFUL = [
    {
        "lat": "40.88939",
        "lon": "-111.88077",
        "display_name": "Bountiful, Davis County, Utah, 84010, United States",
        "address": {
            "city": "Bountiful",
            "county": "Davis County",
            "state": "Utah",
            "postcode": "84010",
            "country": "United States",
            "country_code": "us",
        },
    }
]

_PHOTON_FEATURE_BOUNTIFUL = {
    "features": [
        {
            "geometry": {"coordinates": [-111.88077, 40.88939], "type": "Point"},
            "properties": {
                "name": "Bountiful",
                "city": "Bountiful",
                "state": "Utah",
                "country": "United States",
                "countrycode": "US",
                "postcode": "84010",
            },
        }
    ]
}


@respx.mock
async def test_resolve_address_nominatim_hit_normalises_to_shared_shape():
    # Happy path: Nominatim returns a populated match on the first try.
    # Every field in our normalised shape comes from the nested
    # `address` dict; country_code gets upper-cased to match our
    # ISO-3166 convention downstream.
    respx.get(server.NOMINATIM_URL).mock(
        return_value=httpx.Response(200, json=_NOMINATIM_HIT_BOUNTIFUL)
    )
    r = await server.resolve_address("Bountiful, Utah, 84010")
    assert r["relay_to_user"] is True
    assert r["city"] == "Bountiful"
    assert r["state"] == "Utah"
    assert r["country_code"] == "US"  # upper-cased from "us"
    assert r["postcode"] == "84010"
    assert abs(r["latitude"] - 40.88939) < 1e-6
    assert r["source"] == "nominatim"


@respx.mock
async def test_resolve_address_falls_back_to_photon_when_nominatim_is_empty():
    # Nominatim sometimes returns [] for addresses OSM does tag but
    # Photon indexes differently. Fall-through must populate the
    # same shape but flag `source: "photon"` so operators can tell
    # from logs which path served the hit.
    respx.get(server.NOMINATIM_URL).mock(return_value=httpx.Response(200, json=[]))
    respx.get(server.PHOTON_URL).mock(
        return_value=httpx.Response(200, json=_PHOTON_FEATURE_BOUNTIFUL)
    )
    r = await server.resolve_address("Bountiful, Utah, 84010")
    assert r["relay_to_user"] is True
    assert r["source"] == "photon"
    assert r["city"] == "Bountiful"
    assert r["country_code"] == "US"


@respx.mock
async def test_resolve_address_falls_back_to_photon_when_nominatim_errors():
    # 500 on Nominatim should NOT crash — drop through to Photon.
    respx.get(server.NOMINATIM_URL).mock(return_value=httpx.Response(503))
    respx.get(server.PHOTON_URL).mock(
        return_value=httpx.Response(200, json=_PHOTON_FEATURE_BOUNTIFUL)
    )
    r = await server.resolve_address("Bountiful, Utah, 84010")
    assert r["source"] == "photon"


@respx.mock
async def test_resolve_address_retries_nominatim_with_script_language_as_last_hop():
    # Hop 3: OSM has a Japanese-only name tag for the place, so
    # Nominatim-English returns empty AND Photon returns empty too.
    # The retry with `accept-language=ja` finally finds it. Covers
    # the "native-script-only place" failure mode.
    ja_hit = [{
        "lat": "35.44",
        "lon": "139.65",
        "display_name": "横浜市, 神奈川県, 日本",
        "address": {
            "city": "横浜市",
            "state": "神奈川県",
            "country": "日本",
            "country_code": "jp",
        },
    }]
    call_count = {"n": 0}

    def nominatim_handler(request):
        call_count["n"] += 1
        accept_lang = request.headers.get("Accept-Language", "")
        # First hop: accept-language=en → empty. Later hop with "ja"
        # → populated.
        if accept_lang == "en":
            return httpx.Response(200, json=[])
        if "ja" in accept_lang.lower():
            return httpx.Response(200, json=ja_hit)
        return httpx.Response(200, json=[])

    respx.get(server.NOMINATIM_URL).mock(side_effect=nominatim_handler)
    respx.get(server.PHOTON_URL).mock(return_value=httpx.Response(200, json={"features": []}))

    r = await server.resolve_address("横浜市中区")
    assert r["relay_to_user"] is True
    assert r["city"] == "横浜市"
    assert r["country_code"] == "JP"
    assert call_count["n"] >= 2  # at least en + ja


@respx.mock
async def test_resolve_address_returns_relay_to_user_false_when_all_sources_empty():
    # No source has a match. The tool MUST NOT silently lie with a
    # plausible-looking but wrong hit; instead it returns
    # relay_to_user=false + guidance telling the LLM to ask the
    # user to rephrase.
    respx.get(server.NOMINATIM_URL).mock(return_value=httpx.Response(200, json=[]))
    respx.get(server.PHOTON_URL).mock(return_value=httpx.Response(200, json={"features": []}))
    r = await server.resolve_address("zzz nowhere zzz")
    assert r["relay_to_user"] is False
    assert "rephrase" in r["guidance"].lower() or "ask the user" in r["guidance"].lower()
    assert r["candidates"] == []


@respx.mock
async def test_resolve_address_sends_nominatim_user_agent_per_osm_tos():
    # Nominatim ToS requires a descriptive User-Agent. We share the
    # same UA with radio-browser since it's one repo. This test
    # pins that contract — removing the UA would be a silent ToS
    # violation.
    captured = {}

    def handler(request):
        captured["ua"] = request.headers.get("User-Agent", "")
        captured["lang"] = request.headers.get("Accept-Language", "")
        return httpx.Response(200, json=_NOMINATIM_HIT_BOUNTIFUL)

    respx.get(server.NOMINATIM_URL).mock(side_effect=handler)
    await server.resolve_address("Bountiful, Utah, 84010")
    assert "mcp-weather-simple" in captured["ua"]
    assert captured["lang"] == "en"


async def test_resolve_address_rejects_empty_input():
    with pytest.raises(ValueError, match="non-empty"):
        await server.resolve_address("   ")


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


# ── Bundled free-ride fields (sunrise/sunset/feels_like/uv/precip_prob) ──
#
# Several high-frequency follow-ups — "во сколько закат?", "по ощущениям
# сколько?", "UV высокий?", "зонтик нужен?" — used to cost a second
# round-trip (`get_sunrise_sunset`, etc.). They all come from the SAME
# Open-Meteo /v1/forecast call, so the current-weather + daily-forecast
# impls now ask for them up-front and surface them in the response body.
# Docstring stays silent on purpose (self-explanatory field names keep
# the catalog token count flat); these tests are the written contract.


@respx.mock
async def test_current_weather_bundles_sunrise_uv_precip_prob():
    respx.get(server.GEOCODE_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {"name": "Kyiv", "country": "Ukraine", "country_code": "UA",
                     "latitude": 50.45, "longitude": 30.52, "timezone": "Europe/Kyiv",
                     "feature_code": "PPLC", "admin1": "Kyiv City",
                     "population": 3_000_000},
                ]
            },
        )
    )
    captured: dict[str, str] = {}

    def _respond(req):
        captured.update(dict(req.url.params))
        return httpx.Response(
            200,
            json={
                "current": {
                    "time": "2026-04-21T10:00",
                    "temperature_2m": 12.3,
                    "relative_humidity_2m": 60,
                    "apparent_temperature": 9.5,
                    "precipitation": 0.0,
                    "precipitation_probability": 40,
                    "weather_code": 2,
                    "wind_speed_10m": 10.0,
                    "wind_direction_10m": 180,
                    "uv_index": 3.7,
                },
                "daily": {
                    "time": ["2026-04-21"],
                    "sunrise": ["2026-04-21T05:45"],
                    "sunset":  ["2026-04-21T20:12"],
                },
            },
        )

    respx.get(server.FORECAST_URL).mock(side_effect=_respond)

    r = await server.get_current_weather_in_city("Kyiv", country_code="UA")

    # The request actually asks for the new variables — prevents silent
    # removal of a CSV entry from breaking the contract without a test
    # signal.
    assert "precipitation_probability" in captured["current"]
    assert "uv_index" in captured["current"]
    assert "sunrise" in captured["daily"] and "sunset" in captured["daily"]
    assert captured["forecast_days"] == "1"

    assert r["relay_to_user"] is True
    assert r["apparent_temperature_c"] == 9.5
    assert r["humidity_pct"] == 60
    assert r["precipitation_probability_pct"] == 40
    assert r["uv_index"] == 3.7
    assert r["sunrise"] == "2026-04-21T05:45"
    assert r["sunset"] == "2026-04-21T20:12"


@respx.mock
async def test_current_weather_handles_missing_bundled_fields_gracefully():
    # Open-Meteo has, in the past, silently dropped a variable when the
    # lat/lon falls outside the coverage area (e.g. UV index over
    # far-north winter). The tool must degrade to `null` rather than
    # crashing the whole turn.
    respx.get(server.GEOCODE_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {"name": "Svalbard", "country": "Norway", "country_code": "NO",
                     "latitude": 78.2, "longitude": 15.6, "timezone": "Europe/Oslo",
                     "feature_code": "PPL"},
                ]
            },
        )
    )
    respx.get(server.FORECAST_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "current": {
                    "time": "2026-12-21T10:00",
                    "temperature_2m": -18.0,
                    "relative_humidity_2m": 80,
                    "apparent_temperature": -25.0,
                    "precipitation": 0.0,
                    "weather_code": 1,
                    "wind_speed_10m": 20.0,
                    "wind_direction_10m": 90,
                    # precipitation_probability / uv_index intentionally absent
                },
                # daily intentionally absent — polar night has no sunset
            },
        )
    )

    r = await server.get_current_weather_in_city("Svalbard")
    assert r["relay_to_user"] is True
    assert r["uv_index"] is None
    assert r["precipitation_probability_pct"] is None
    assert r["sunrise"] is None
    assert r["sunset"] is None


@respx.mock
async def test_daily_forecast_bundles_feels_like_uv_and_sun_times():
    respx.get(server.GEOCODE_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {"name": "Paris", "country": "France", "country_code": "FR",
                     "latitude": 48.85, "longitude": 2.35, "timezone": "Europe/Paris",
                     "feature_code": "PPLC"},
                ]
            },
        )
    )
    captured: dict[str, str] = {}

    def _respond(req):
        captured.update(dict(req.url.params))
        return httpx.Response(
            200,
            json={
                "daily": {
                    "time": ["2026-04-22", "2026-04-23"],
                    "weather_code": [2, 3],
                    "temperature_2m_max": [22.0, 24.0],
                    "temperature_2m_min": [10.0, 11.0],
                    "apparent_temperature_max": [20.5, 22.5],
                    "apparent_temperature_min": [8.5, 9.5],
                    "precipitation_sum": [0.0, 2.0],
                    "precipitation_probability_max": [15, 70],
                    "wind_speed_10m_max": [12.0, 20.0],
                    "uv_index_max": [5.2, 4.8],
                    "sunrise": ["2026-04-22T06:45", "2026-04-23T06:43"],
                    "sunset":  ["2026-04-22T20:58", "2026-04-23T21:00"],
                },
            },
        )

    respx.get(server.FORECAST_URL).mock(side_effect=_respond)

    r = await server.get_weather_forecast("Paris", days=2, country_code="FR")

    assert "apparent_temperature_max" in captured["daily"]
    assert "uv_index_max" in captured["daily"]
    assert "sunrise" in captured["daily"] and "sunset" in captured["daily"]

    today, tomorrow = r["days"]
    assert today["feels_like_max_c"] == 20.5
    assert today["feels_like_min_c"] == 8.5
    assert today["uv_index_max"] == 5.2
    assert today["sunrise"] == "2026-04-22T06:45"
    assert today["sunset"] == "2026-04-22T20:58"
    assert tomorrow["feels_like_max_c"] == 22.5
    assert tomorrow["sunset"] == "2026-04-23T21:00"


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
    # The old `accuracy_warning` body field is subsumed by #18's
    # `guidance` envelope — the uncertainty is surfaced as an
    # instruction to the LLM, not an optional body hint.
    assert result["relay_to_user"] is True
    assert "caveat" in result["guidance"].lower()


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
async def test_wikipedia_summary_falls_back_from_en_to_ru_for_cyrillic_title():
    # Live mcphost session on 2026-04-21: 14b called
    # `get_wikipedia_summary(title="соль")` with default lang="en".
    # en.wiki → 404 (no article titled «соль» in English). Without a
    # lang fallback the tool errored out; with the script-based chain
    # we silently retry on ru.wiki and get the real article.
    respx.get(
        server.WIKIPEDIA_SUMMARY_URL.format(lang="en", title="соль")
    ).mock(return_value=httpx.Response(404, json={"type": "not-found"}))
    respx.get(
        server.WIKIPEDIA_SUMMARY_URL.format(lang="ru", title="соль")
    ).mock(
        return_value=httpx.Response(
            200,
            json={
                "title": "Соль",
                "description": "химическое соединение",
                "extract": "Соль (хлорид натрия) — вещество с формулой NaCl.",
                "content_urls": {"desktop": {"page": "https://ru.wikipedia.org/wiki/Соль"}},
            },
        )
    )

    r = await server.get_wikipedia_summary("соль")
    assert r["lang"] == "ru"
    assert r["title"] == "Соль"
    # chain reports ALL languages tried (even if only ru succeeded) —
    # useful debug signal for operators.
    assert "en" in r["lang_chain_tried"]
    assert "ru" in r["lang_chain_tried"]


@respx.mock
async def test_wikipedia_summary_explicit_lang_wins_over_script_fallback():
    # When the caller explicitly passes lang="de", Wikipedia should
    # hit de.wiki FIRST — the script-detected chain only kicks in if
    # the primary fails. Here de.wiki returns a hit immediately, so
    # we never fall back.
    de_hit = httpx.Response(
        200,
        json={
            "title": "Berlin",
            "extract": "Berlin ist die Hauptstadt Deutschlands.",
            "content_urls": {"desktop": {"page": "https://de.wikipedia.org/wiki/Berlin"}},
        },
    )
    de_mock = respx.get(
        server.WIKIPEDIA_SUMMARY_URL.format(lang="de", title="Berlin")
    ).mock(return_value=de_hit)
    # If the fallback was incorrectly triggered, en.wiki would be
    # hit too — this mock stays unused on a clean run.
    respx.get(
        server.WIKIPEDIA_SUMMARY_URL.format(lang="en", title="Berlin")
    ).mock(return_value=httpx.Response(200, json={"title": "UnusedEnglishHit"}))

    r = await server.get_wikipedia_summary("Berlin", lang="de")
    assert r["lang"] == "de"
    assert r["title"] == "Berlin"
    # First-tried language is the caller's explicit choice.
    assert r["lang_chain_tried"][0] == "de"
    assert de_mock.called


@respx.mock
async def test_wikipedia_summary_raises_when_every_lang_misses():
    # All candidate langs return 404 → we re-raise the last upstream
    # error so the LLM sees "Wikipedia rejected HTTP 404" and can
    # surface it to the user, not a silent empty response.
    for lang in ("en", "ru", "uk"):
        respx.get(
            server.WIKIPEDIA_SUMMARY_URL.format(lang=lang, title="несуществующаястатья42")
        ).mock(return_value=httpx.Response(404, json={"type": "not-found"}))
    with pytest.raises(RuntimeError, match="HTTP 404"):
        await server.get_wikipedia_summary("несуществующаястатья42")


@respx.mock
async def test_wikipedia_summary_sends_descriptive_user_agent():
    # Wikipedia REST API enforces its User-Agent ToS — a request
    # without a descriptive UA gets HTTP 403. Discovered live on
    # 2026-04-21 when a mcphost session tried `get_wikipedia_summary`
    # and hit 403 from real Wikipedia. Pin the contract here so a
    # refactor doesn't silently drop the header again.
    captured = {}

    def handler(request):
        captured["ua"] = request.headers.get("User-Agent", "")
        return httpx.Response(
            200,
            json={
                "title": "Bountiful",
                "extract": "…",
                "content_urls": {"desktop": {"page": "https://en.wikipedia.org/wiki/Bountiful"}},
            },
        )

    respx.get(
        server.WIKIPEDIA_SUMMARY_URL.format(lang="en", title="Bountiful")
    ).mock(side_effect=handler)
    await server.get_wikipedia_summary("Bountiful")
    assert "mcp-weather-simple" in captured["ua"]
    assert "github.com" in captured["ua"]


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


async def _probe(app, path: str, *, headers: dict | None = None) -> httpx.Response:
    # Drive the Starlette/FastMCP app through an in-memory ASGI
    # transport — no port, no uvicorn, works in parallel pytest.
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://mcp.test") as client:
        return await client.get(path, headers=headers or {})


async def test_liveness_paths_return_200_without_io():
    # /healthz and /livez are the LIVENESS probes — no I/O at all,
    # always 200. If we ever start doing upstream work in this path
    # a hung dependency could flip liveness and cause a pod restart
    # loop.
    with patch.object(server, "AUTH_TOKEN", ""):
        app = server._build_http_app()
    for path in ("/healthz", "/livez"):
        r = await _probe(app, path)
        assert r.status_code == 200, f"{path} should be 200, got {r.status_code}"
        body = r.json()
        assert body["status"] == "ok"
        assert body["service"] == "mcp-weather"
        assert body["probe"] == "liveness"


@respx.mock
async def test_readiness_probe_returns_200_when_upstream_is_reachable():
    # /readyz actively probes Open-Meteo's geocoder. On success the
    # response names the dependency so operators can tell at a glance
    # WHICH check passed.
    respx.get(server.GEOCODE_URL).mock(
        return_value=httpx.Response(200, json={"results": []})
    )
    with patch.object(server, "AUTH_TOKEN", ""):
        app = server._build_http_app()
    r = await _probe(app, "/readyz")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["probe"] == "readiness"
    assert body["checks"] == {"open_meteo": "ok"}


@respx.mock
async def test_readiness_probe_returns_503_when_upstream_times_out():
    # When Open-Meteo is unreachable every real tool call will fail
    # the same way — /readyz must surface that to k8s instead of
    # pretending to be ready. 503 + status=not_ready + the exception
    # class name so operators can grep `ReadTimeout` in pod logs.
    respx.get(server.GEOCODE_URL).mock(side_effect=httpx.ReadTimeout("slow"))
    with patch.object(server, "AUTH_TOKEN", ""):
        app = server._build_http_app()
    r = await _probe(app, "/readyz")
    assert r.status_code == 503
    body = r.json()
    assert body["status"] == "not_ready"
    assert body["probe"] == "readiness"
    assert "open_meteo" in body["checks"]
    assert "ReadTimeout" in body["checks"]["open_meteo"]


@respx.mock
async def test_probes_bypass_bearer_auth_when_token_is_set():
    # With a token configured, the MCP paths require bearer; probes
    # do not — k8s probes never carry a secret and must still reach
    # both liveness AND readiness endpoints.
    respx.get(server.GEOCODE_URL).mock(
        return_value=httpx.Response(200, json={"results": []})
    )
    with patch.object(server, "AUTH_TOKEN", "secret-sentinel"):
        app = server._build_http_app()

    live = await _probe(app, "/livez")
    assert live.status_code == 200

    ready = await _probe(app, "/readyz")
    assert ready.status_code == 200

    # And that the bearer middleware is actually armed: a GET to an
    # unknown path without the header returns 401, not 200.
    r_unauth = await _probe(app, "/mcp")
    assert r_unauth.status_code == 401


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
async def test_detect_my_location_by_ip_takes_no_args_and_uses_autodetect_guidance():
    # After the split, `detect_my_location_by_ip` has no `ip` param —
    # the intent is strictly "where am I?". Guidance is the autodetect
    # caveat (IP may be wrong behind VPN / NAT / cluster egress).
    # Addresses come from RFC 5737 reserved "documentation" ranges
    # (TEST-NET-2 / TEST-NET-3) — never routed on the public internet.
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
    r = await server.detect_my_location_by_ip()
    assert r["location_source"] == "geoip_autodetected"
    assert r["guidance"] == server._GUIDANCE_GEOIP_AUTODETECT
    assert r["city"] == "Auto"


@respx.mock
async def test_lookup_ip_geolocation_requires_explicit_ip_and_uses_explicit_guidance():
    # New sibling tool — ip is REQUIRED. Guidance is the explicit-IP
    # caveat (IP-to-geo DB depends on which network the IP belongs to).
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
    r = await server.lookup_ip_geolocation(ip="198.51.100.7")
    assert r["location_source"] == "geoip_explicit"
    assert r["guidance"] == server._GUIDANCE_GEOIP_EXPLICIT
    assert r["city"] == "Manual"


async def test_lookup_ip_geolocation_rejects_empty_ip():
    with pytest.raises(ValueError, match="ip argument is required"):
        await server.lookup_ip_geolocation(ip="")
    with pytest.raises(ValueError, match="ip argument is required"):
        await server.lookup_ip_geolocation(ip="   ")


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
    assert r["data_source"] == "rest_countries"


@respx.mock
async def test_country_info_falls_back_to_graphql_mirror_on_rest_5xx():
    """restcountries.com is volunteer-hosted and 502s regularly
    (hit twice in the chat UI on 2026-04-23). Fallback to the
    trevorblades GraphQL mirror must kick in and return the
    common subset — no population / area / borders / timezones,
    but capital / currency / languages / emoji are all present.
    `data_source` flips to `graphql_mirror` so the model can caveat.
    """
    respx.get(server.RESTCOUNTRIES_ALPHA_URL.format(code="CN")).mock(
        return_value=httpx.Response(502)
    )
    respx.post(server.COUNTRIES_GRAPHQL_URL).mock(
        return_value=httpx.Response(200, json={
            "data": {"country": {
                "code": "CN",
                "name": "China",
                "native": "中国",
                "capital": "Beijing",
                "emoji": "🇨🇳",
                "phone": "86",
                "currency": "CNY",
                "continent": {"name": "Asia"},
                "languages": [{"name": "Chinese"}],
            }}
        })
    )
    r = await server.get_country_info("CN")
    assert r["data_source"] == "graphql_mirror"
    assert r["name"] == "China"
    assert r["capital"] == "Beijing"
    assert r["calling_code"] == "+86"
    assert r["currencies"] == [{"code": "CNY", "name": None, "symbol": None}]
    assert r["languages"] == ["Chinese"]
    # Fields not covered by the mirror degrade to null, not missing.
    assert r["population"] is None
    assert r["area_km2"] is None
    assert r["borders"] == []
    # Guidance must explicitly warn about the narrower field set.
    assert "mirror" in r["guidance"].lower()


@respx.mock
async def test_country_info_reraises_when_both_primary_and_mirror_fail():
    """If restcountries 5xx AND the GraphQL mirror also fails (network
    down, rate-limit, etc.), surface the ORIGINAL restcountries error.
    Using the mirror error would be misleading — the primary being
    down is the actual story the user should get."""
    respx.get(server.RESTCOUNTRIES_ALPHA_URL.format(code="CN")).mock(
        return_value=httpx.Response(502)
    )
    respx.post(server.COUNTRIES_GRAPHQL_URL).mock(
        return_value=httpx.Response(503)  # mirror also sick
    )
    with pytest.raises(RuntimeError, match="REST Countries"):
        await server.get_country_info("CN")


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


# ── calculate (safe AST walker) ───────────────────────────────────────────


@pytest.mark.parametrize(
    "expr,expected",
    [
        ("2 + 3", 5),
        ("3847 * 29", 111563),                        # 4-digit × 2-digit
        ("2450 * 0.15", 367.5),                       # percentage
        ("(300 + 50) * 1.08 / 2", 189.0),             # chained
        ("2 ** 10", 1024),                            # power
        ("pow(2, 10)", 1024),                         # builtin pow
        ("hypot(3, 4)", 5.0),                         # geometry primitive
        ("abs(-42)", 42),
        ("round(3.14159, 2)", 3.14),
        ("min(3, 1, 2)", 1),
        ("max(3, 1, 2)", 3),
        ("factorial(5)", 120),
        ("gcd(12, 18)", 6),
        ("-5 + 3", -2),                               # unary
        ("+5", 5),
    ],
)
async def test_calculate_happy_path(expr, expected):
    import server as srv
    r = await srv.calculate(expr)
    assert r.get("error") is None, f"unexpected error on {expr!r}: {r}"
    assert r["expression"] == expr
    assert abs(r["result"] - expected) < 1e-9 if isinstance(expected, float) else r["result"] == expected


async def test_calculate_circle_area_via_pi_constant():
    import math
    import server as srv
    r = await srv.calculate("pi * 5**2")
    assert r["result"] == pytest.approx(math.pi * 25, rel=1e-12)


async def test_calculate_trig_via_radians():
    import math
    import server as srv
    r = await srv.calculate("sin(radians(30))")
    assert r["result"] == pytest.approx(0.5, rel=1e-9)


@pytest.mark.parametrize(
    "expr,expected",
    [
        ("2^10",              1024),                # plain power
        ("pi * 5^2",          3.141592653589793 * 25),  # circle area — natural notation
        ("pi * 0^2",          0.0),                 # the live-reported case from the chat UI
        ("(4/3) * pi * 2^3",  (4/3) * 3.141592653589793 * 8),  # sphere volume
        ("2^3^2",             512),                 # right-associativity: 2^(3^2) = 2^9
        ("3 + 4^2",           19),                  # precedence: 4^2 binds tighter than +
    ],
)
async def test_calculate_caret_is_power_shortcut(expr, expected):
    """`^` is rewritten to `**` before parsing so LLMs writing natural
    math notation (`pi * r^2`, `2^10`) don't get BitXor rejections.
    Precedence must match math convention — `**` binds tighter than
    `*`/`+`, so `pi * r^2` means `pi * (r^2)`, not `(pi*r)^2`. Triggered
    by a live report: model called `calculate("pi * 0^2")` for a
    zero-radius circle and hit `operator not allowed: BitXor`."""
    import server as srv
    r = await srv.calculate(expr)
    assert r.get("error") is None, f"unexpected error on {expr!r}: {r}"
    if isinstance(expected, float):
        assert r["result"] == pytest.approx(expected, rel=1e-12)
    else:
        assert r["result"] == expected


@pytest.mark.parametrize(
    "expr,error_fragment",
    [
        ("foo + 1",                      "unknown name"),           # identifier
        ("math.sqrt(2)",                 "attribute access"),       # attribute
        ("__import__('os')",             "unknown function"),       # dunder
        ("open('/etc/passwd')",          "unknown function"),       # builtin fn not in whitelist
        ("x = 5",                         "syntax error"),           # assignment
        ("[1, 2, 3]",                    "expression node not allowed"),  # list
        ("{'a': 1}",                     "expression node not allowed"),  # dict
        ("'hello'",                      "only numeric literals"),  # string
        ("1 if True else 2",             "expression node not allowed"),  # ternary
        ("lambda x: x",                  "expression node not allowed"),  # lambda
    ],
)
async def test_calculate_rejects_unsafe_expressions(expr, error_fragment):
    """Every non-arithmetic AST shape must be rejected with a clear
    message. The rejections are the whole security story of this tool:
    no `eval()`, no attribute access, no imports, no names outside the
    constants whitelist. Regression guard against accidentally adding
    an escape hatch to the walker later."""
    import server as srv
    r = await srv.calculate(expr)
    assert "error" in r, f"expected error for {expr!r}, got {r}"
    assert error_fragment in r["error"], (
        f"error message for {expr!r} ({r['error']!r}) should mention {error_fragment!r}"
    )
    # Refusals disable relay — the model must not fabricate a result.
    assert r["relay_to_user"] is False


async def test_calculate_division_by_zero_is_handled_gracefully():
    import server as srv
    r = await srv.calculate("1 / 0")
    assert "division by zero" in r["error"]
    assert r["relay_to_user"] is False


# ── Router prototype (MCP_ROUTER_MODE=list_changed) ───────────────────────


@pytest.mark.asyncio
async def test_router_mode_filters_list_tools_by_active_domain(monkeypatch):
    """Router mode is opt-in via env; off by default.

    Exercises the three states the filtered `list_tools` handler has to
    get right:
      1. no domain selected → only `select_domain` is visible,
      2. domain selected    → `select_domain` + that domain's tools,
      3. domain switched    → the OTHER domain's tools (not both).
    """
    import importlib

    import server as srv

    monkeypatch.setenv("MCP_ROUTER_MODE", "list_changed")
    srv = importlib.reload(srv)
    try:
        assert srv.ROUTER_MODE == "list_changed"
        assert srv.mcp.settings.stateless_http is False

        from mcp.types import ListToolsRequest

        handler = srv.mcp._mcp_server.request_handlers[ListToolsRequest]
        req = ListToolsRequest(method="tools/list")

        # State 1: fresh session, no domain picked yet.
        srv._ROUTER_STATE["active"] = None
        names = sorted(t.name for t in (await handler(req)).root.tools)
        assert names == ["select_domain"]

        # State 2: weather picked → weather tools + router entry.
        srv._ROUTER_STATE["active"] = "weather"
        names = sorted(t.name for t in (await handler(req)).root.tools)
        assert "select_domain" in names
        assert "get_current_weather_in_city" in names
        # Must not leak tools from other domains.
        assert "get_wikipedia_summary" not in names
        assert "convert_currency" not in names

        # State 3: switch to knowledge — weather-only tools drop.
        srv._ROUTER_STATE["active"] = "knowledge"
        names = sorted(t.name for t in (await handler(req)).root.tools)
        assert "get_wikipedia_summary" in names
        assert "get_current_weather_in_city" not in names
    finally:
        monkeypatch.delenv("MCP_ROUTER_MODE", raising=False)
        importlib.reload(srv)


def test_router_mode_explicit_off_leaves_full_monolith(monkeypatch):
    """Explicit `MCP_ROUTER_MODE=off` restores the full narrow-tool
    monolith with stateless HTTP. Used by the nightly eval cron to
    keep baseline trend data comparable to pre-router numbers."""
    import importlib

    import server as srv

    monkeypatch.setenv("MCP_ROUTER_MODE", "off")
    monkeypatch.delenv("MCP_ENABLED_DOMAINS", raising=False)
    srv = importlib.reload(srv)
    assert srv.ROUTER_MODE == "off"
    assert srv.mcp.settings.stateless_http is True
    names = {t.name for t in srv.mcp._tool_manager.list_tools()}
    assert "select_domain" not in names
    # 24 base narrow tools + 4 web-domain narrows (`web_search`, `news`,
    # `hackernews`, `trends`) added 2026-04-22.
    assert len(names) == 28


def test_router_mode_default_is_fat_tools_lean(monkeypatch):
    """When MCP_ROUTER_MODE isn't set, server boots in fat_tools_lean.

    Flipped 2026-04-22; current catalog (post-web-domain, post-radio-
    fold-in 2026-04-23) ≈ 1350 tokens vs ~6800 for the monolith.
    Hit rate stays at 93.2 % on qwen3.5:9b across live modes. Production
    Docker image + platform sidecar inherit this default so i7-CPU
    deployments stop paying the monolith prefill.
    """
    import importlib

    import server as srv

    monkeypatch.delenv("MCP_ROUTER_MODE", raising=False)
    srv = importlib.reload(srv)
    assert srv.ROUTER_MODE == "fat_tools_lean", (
        f"default should be fat_tools_lean, got {srv.ROUTER_MODE!r}"
    )
    # fat modes register the 4 fat tools in addition to the 28 narrow
    # ones; list_tools filter is what hides the narrow set from clients.
    # Radio folded from its own `radio` fat into `web(radio)` 2026-04-23.
    names = {t.name for t in srv.mcp._tool_manager.list_tools()}
    assert {"weather", "geo", "knowledge", "web"}.issubset(names)
    assert "radio" not in names
    assert len(names) == 32


@pytest.mark.asyncio
async def test_router_mode_fat_tools_exposes_four_domain_tools(monkeypatch):
    """`MCP_ROUTER_MODE=fat_tools` registers weather/geo/knowledge/radio
    and hides every narrow @mcp.tool from `tools/list`.

    The 4 fat tools are the *only* surface the client sees; the 23
    narrow tools remain in the manager but are filtered out so the
    model's catalog drops from ~5300 to ~1900 tokens.
    """
    import importlib

    import server as srv

    monkeypatch.setenv("MCP_ROUTER_MODE", "fat_tools")
    monkeypatch.delenv("MCP_ENABLED_DOMAINS", raising=False)
    srv = importlib.reload(srv)
    try:
        assert srv.ROUTER_MODE == "fat_tools"

        # Manager holds both 28 narrow + 4 fat — filtering is done at
        # list_tools time, not at registration. (Was 5 fat before
        # radio got folded into web 2026-04-23.)
        all_names = {t.name for t in srv.mcp._tool_manager.list_tools()}
        assert {"weather", "geo", "knowledge", "web"}.issubset(all_names)
        assert "get_current_weather_in_city" in all_names  # still registered
        assert len(all_names) == 32

        from mcp.types import ListToolsRequest

        handler = srv.mcp._mcp_server.request_handlers[ListToolsRequest]
        req = ListToolsRequest(method="tools/list")
        visible = sorted(t.name for t in (await handler(req)).root.tools)
        assert visible == ["geo", "knowledge", "weather", "web"]
    finally:
        monkeypatch.delenv("MCP_ROUTER_MODE", raising=False)
        importlib.reload(srv)


@pytest.mark.asyncio
async def test_router_mode_fat_tools_lean_exposes_four_with_params_dict(monkeypatch):
    """`MCP_ROUTER_MODE=fat_tools_lean` registers the same 4 fat names
    but each takes only `(action, params)` — schema shrinks ~50% vs
    plain fat_tools because the per-kwarg nullability shell goes away.
    """
    import importlib

    import server as srv

    monkeypatch.setenv("MCP_ROUTER_MODE", "fat_tools_lean")
    monkeypatch.delenv("MCP_ENABLED_DOMAINS", raising=False)
    srv = importlib.reload(srv)
    try:
        assert srv.ROUTER_MODE == "fat_tools_lean"

        all_names = {t.name for t in srv.mcp._tool_manager.list_tools()}
        assert {"weather", "geo", "knowledge", "web"}.issubset(all_names)
        assert "get_current_weather_in_city" in all_names  # still registered

        from mcp.types import ListToolsRequest
        handler = srv.mcp._mcp_server.request_handlers[ListToolsRequest]
        req = ListToolsRequest(method="tools/list")
        tools = (await handler(req)).root.tools
        assert sorted(t.name for t in tools) == ["geo", "knowledge", "weather", "web"]

        # The lean schema has exactly two top-level props (action, params)
        # on the enum-bearing tools — that's the whole point of the variant.
        weather = next(t for t in tools if t.name == "weather")
        assert set(weather.inputSchema.get("properties", {}).keys()) == {"action", "params"}
    finally:
        monkeypatch.delenv("MCP_ROUTER_MODE", raising=False)
        importlib.reload(srv)


@pytest.mark.asyncio
async def test_fat_tools_lean_weather_dispatch_via_params_dict(monkeypatch):
    """Smoke the lean `weather` dispatcher — `params` dict flows through
    to the narrow impl with required / optional keys honoured."""
    import importlib

    import server as srv

    monkeypatch.setenv("MCP_ROUTER_MODE", "fat_tools_lean")
    srv = importlib.reload(srv)
    try:
        import fat_tools_lean

        sentinel = {"ok": "delegated"}

        async def fake_impl(city, country_code=None):
            assert city == "Kyiv" and country_code == "UA"
            return sentinel

        monkeypatch.setattr(srv, "get_current_weather_in_city", fake_impl)
        result = await fat_tools_lean.weather(
            action="current_in_city", params={"city": "Kyiv", "country_code": "UA"}
        )
        assert result is sentinel

        # Missing required param raises with a clear message.
        with pytest.raises(ValueError, match="missing required"):
            await fat_tools_lean.weather(action="current_in_city", params={})

        # Unknown action fails closed.
        with pytest.raises(ValueError, match="unknown action"):
            await fat_tools_lean.weather(action="does_not_exist", params={})  # type: ignore[arg-type]
    finally:
        monkeypatch.delenv("MCP_ROUTER_MODE", raising=False)
        importlib.reload(srv)


@pytest.mark.asyncio
async def test_fat_weather_dispatches_to_underlying_impl(monkeypatch):
    """Smoke-test the dispatcher in `fat_tools.weather` — confirms the
    `action` enum flows to the correct narrow implementation."""
    import importlib

    import server as srv

    monkeypatch.setenv("MCP_ROUTER_MODE", "fat_tools")
    srv = importlib.reload(srv)
    try:
        import fat_tools

        # Patch out the narrow impl to verify the dispatcher reaches it.
        sentinel = {"ok": "delegated"}

        async def fake_impl(city, country_code=None):
            assert city == "Kyiv"
            return sentinel

        monkeypatch.setattr(srv, "get_current_weather_in_city", fake_impl)
        result = await fat_tools.weather(action="current_in_city", city="Kyiv")
        assert result is sentinel

        # Missing required args raise a clear ValueError.
        with pytest.raises(ValueError, match="missing required"):
            await fat_tools.weather(action="current_in_city")  # no city

        # Unknown action fails closed.
        with pytest.raises(ValueError, match="unknown action"):
            await fat_tools.weather(action="does_not_exist")  # type: ignore[arg-type]
    finally:
        monkeypatch.delenv("MCP_ROUTER_MODE", raising=False)
        importlib.reload(srv)


def test_narrow_to_fat_map_covers_every_registered_tool(monkeypatch):
    """`fat_tools_map.NARROW_TO_FAT` must map every narrow @mcp.tool
    in server.py exactly once. Drift here silently breaks the eval
    scorer in fat mode — kill it at test time.

    Since the default router mode is now `fat_tools_lean`, the tool
    manager also contains the 4 fat tool names; pin to `off` here
    so `registered` is the pure narrow set we want to check coverage
    against.
    """
    import importlib

    import fat_tools_map
    import server as srv

    monkeypatch.setenv("MCP_ROUTER_MODE", "off")
    srv = importlib.reload(srv)

    registered = {t.name for t in srv.mcp._tool_manager.list_tools()}
    mapped = set(fat_tools_map.NARROW_TO_FAT.keys())
    missing = registered - mapped
    extra = mapped - registered
    assert not missing, f"NARROW_TO_FAT missing narrow tools: {sorted(missing)}"
    assert not extra, f"NARROW_TO_FAT references unknown tools: {sorted(extra)}"

    # Every mapped fat tool name must be one of the 5 known domains.
    fat_tool_names = {fat for fat, _ in fat_tools_map.NARROW_TO_FAT.values()}
    assert fat_tool_names == {"weather", "geo", "knowledge", "web"}, (
        f"unexpected fat tool name(s): {fat_tool_names}"
    )


@pytest.mark.parametrize(
    "narrow,canonical",
    [
        ("get_current_weather_in_city", "weather(current_in_city)"),
        ("get_wikipedia_summary",       "knowledge(wikipedia)"),
        ("list_radio_stations",         "web(radio)"),
        ("get_current_date",            "geo(date_in_timezone)"),
    ],
)
def test_eval_canonicalise_expected_fat_mode(monkeypatch, narrow, canonical):
    """The eval scorer's `_canonicalise_expected` rewrites narrow tool
    names into `fat(action)` strings (or bare fat name for radio) when
    `MCP_ROUTER_MODE=fat_tools`. Narrow mode passes through unchanged.

    Loading `eval_tool_calling.py` pulls in `yaml` at module scope, which
    only lives in the `eval` extra — the CI unit-test job syncs only
    `--extra test`. Skip here so this test is meaningful locally (where
    `uv sync --extra test --extra eval` is typical) and invisible in CI
    until the unit job starts pulling eval deps too.
    """
    pytest.importorskip("yaml", reason="requires the 'eval' extra (pyyaml)")
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "eval_tc_test",
        Path(__file__).parent.parent / "tests" / "integration" / "eval_tool_calling.py",
    )

    # Both fat router variants share canonical names — a single scorer
    # branch has to handle both. Regression guard against the bug that
    # shipped on 2026-04-22: `_FAT_MODE = mode == "fat_tools"` missed
    # `fat_tools_lean` entirely, every lean-run scored 0/44.
    for mode_value in ("fat_tools", "fat_tools_lean"):
        monkeypatch.setenv("MCP_ROUTER_MODE", mode_value)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        m._load_fat_mapping_if_enabled()
        assert m._FAT_MODE is True, f"mode={mode_value!r} should enable fat scoring"
        assert m._canonicalise_expected(narrow) == canonical

    # Narrow-mode path: reload without the env, canonicalise is a no-op.
    monkeypatch.delenv("MCP_ROUTER_MODE", raising=False)
    m2 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m2)
    m2._load_fat_mapping_if_enabled()
    assert m2._FAT_MODE is False
    assert m2._canonicalise_expected(narrow) == narrow


# ── Web domain (search / news / hackernews / trends) ─────────────────────
#
# Each test mocks the upstream provider at the HTTP boundary so the
# suite stays offline. The goal is both response-shape coverage and
# **prompt-engineering** coverage — we want the fixtures to reflect
# the real provider's quirks (DDG's redirect wrapping, Google News'
# nested <source> tag, HN's two-phase ID-list + per-item fetch,
# Google Trends' namespaced XML extensions) so regressions in the
# parsing code show up immediately.


@respx.mock
async def test_web_search_parses_ddg_lite_html():
    ddg_html = """
    <html><body>
      <div class="result">
        <a rel="nofollow" class="result__a" href="https://duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fkyiv&rut=x">
          Kyiv — Wikipedia
        </a>
        <a class="result__snippet" href="ignored">
          Kyiv is the capital and most populous city of Ukraine.
        </a>
      </div>
      <div class="result">
        <a rel="nofollow" class="result__a" href="https://duckduckgo.com/l/?uddg=https%3A%2F%2Fkyivindependent.com%2F">
          Kyiv Independent
        </a>
        <a class="result__snippet">Independent English-language newsroom.</a>
      </div>
    </body></html>
    """
    respx.get(server.DDG_LITE_URL).mock(return_value=httpx.Response(200, text=ddg_html))

    r = await server.web_search("Kyiv")
    assert r["relay_to_user"] is True
    assert len(r["results"]) == 2
    first = r["results"][0]
    # DDG-redirect wrapper should be unwrapped to the real destination.
    assert first["url"] == "https://example.com/kyiv"
    assert "Kyiv" in first["title"]
    assert "capital" in first["snippet"].lower()


@respx.mock
async def test_web_search_empty_results_sets_relay_false():
    respx.get(server.DDG_LITE_URL).mock(
        return_value=httpx.Response(200, text="<html><body>No hits</body></html>")
    )
    r = await server.web_search("jsklfdjsakl")
    assert r["results"] == []
    assert r["relay_to_user"] is False
    assert "rephrase" in r["guidance"].lower() or "shape" in r["guidance"].lower()


async def test_web_search_rejects_empty_query():
    with pytest.raises(ValueError, match="query is required"):
        await server.web_search("   ")


@respx.mock
async def test_news_no_args_uses_geoip_top_rss():
    """No-args news → GeoIP → Google News top RSS with hl/gl/ceid set."""
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
    captured: dict[str, str] = {}

    def _respond(req):
        captured.update(dict(req.url.params))
        rss = """<?xml version="1.0" encoding="UTF-8"?>
        <rss version="2.0"><channel>
          <item>
            <title>Top story: Kyiv in the news</title>
            <link>https://news.example/1</link>
            <source url="https://example.com">Example News</source>
            <pubDate>Mon, 21 Apr 2026 10:00:00 GMT</pubDate>
            <description>Lorem ipsum &lt;b&gt;dolor&lt;/b&gt;.</description>
          </item>
        </channel></rss>
        """
        return httpx.Response(200, text=rss)

    respx.get(server.GNEWS_TOP_URL).mock(side_effect=_respond)
    r = await server.news()
    # GeoIP-derived params must end up on the upstream call — guards the
    # "top-news is locale-aware" contract.
    assert captured["gl"] == "UA"
    assert captured["ceid"] == "UA:en"
    assert r["shape"] == "top"
    assert r["country"] == "UA"
    assert len(r["items"]) == 1
    assert r["items"][0]["source"] == "Example News"
    # HTML tags in description must be stripped so the model doesn't
    # re-emit them in user-facing prose.
    assert "<" not in r["items"][0]["description"]


@respx.mock
async def test_news_with_query_uses_search_rss():
    captured: dict[str, str] = {}

    def _respond(req):
        captured.update(dict(req.url.params))
        rss = """<?xml version="1.0"?>
        <rss><channel>
          <item>
            <title>OpenAI ships something</title>
            <link>https://example/openai</link>
            <source>Example Tech</source>
            <pubDate>Tue, 22 Apr 2026 09:00:00 GMT</pubDate>
          </item>
        </channel></rss>
        """
        return httpx.Response(200, text=rss)

    respx.get(server.GNEWS_SEARCH_URL).mock(side_effect=_respond)

    r = await server.news(query="OpenAI", lang="en")
    assert captured["q"] == "OpenAI"
    assert captured["hl"] == "en"
    assert r["shape"] == "query"
    assert r["items"][0]["title"].startswith("OpenAI")


async def test_news_rejects_query_and_topic_together():
    with pytest.raises(ValueError, match="either"):
        await server.news(query="x", topic="y")


@respx.mock
async def test_hackernews_fetches_ids_then_items_in_parallel():
    respx.get(f"{server.HN_API_BASE}/topstories.json").mock(
        return_value=httpx.Response(200, json=[101, 102, 103])
    )
    respx.get(f"{server.HN_API_BASE}/item/101.json").mock(
        return_value=httpx.Response(200, json={
            "id": 101, "title": "A thing", "url": "https://a.example",
            "score": 120, "descendants": 45, "by": "alice", "time": 1_700_000_000,
        })
    )
    respx.get(f"{server.HN_API_BASE}/item/102.json").mock(
        return_value=httpx.Response(200, json={
            "id": 102, "title": "Ask HN: X?", "score": 30, "descendants": 12,
            "by": "bob", "time": 1_700_000_100,
            # No `url` — typical Ask HN item. Tool should fall back to hn_url.
        })
    )
    respx.get(f"{server.HN_API_BASE}/item/103.json").mock(
        return_value=httpx.Response(200, json={
            "id": 103, "title": "Show HN: Y", "url": "https://y.example",
            "score": 80, "descendants": 20, "by": "carol", "time": 1_700_000_200,
        })
    )

    r = await server.hackernews("top", limit=3)
    assert r["category"] == "top"
    assert [it["title"] for it in r["items"]] == ["A thing", "Ask HN: X?", "Show HN: Y"]
    # Ask HN's item has no url → falls back to the HN item page.
    ask_hn = r["items"][1]
    assert ask_hn["url"] == "https://news.ycombinator.com/item?id=102"
    assert ask_hn["hn_url"] == "https://news.ycombinator.com/item?id=102"


async def test_hackernews_rejects_unknown_category():
    with pytest.raises(ValueError, match="unknown HN category"):
        await server.hackernews("nonsense")


@respx.mock
async def test_trends_defaults_country_via_geoip():
    respx.get(f"{server.GEOIP_URL}/").mock(
        return_value=httpx.Response(
            200,
            json={
                "success": True, "ip": "203.0.113.10", "city": "Salt Lake City",
                "country": "United States", "country_code": "US",
                "latitude": 40.76, "longitude": -111.89,
                "timezone": {"id": "America/Denver"},
            },
        )
    )
    captured: dict[str, str] = {}

    def _respond(req):
        captured.update(dict(req.url.params))
        rss = """<?xml version="1.0" encoding="UTF-8"?>
        <rss xmlns:ht="https://trends.google.com/trending/rss" version="2.0">
          <channel>
            <item>
              <title>Mars mission</title>
              <ht:approx_traffic>500,000+</ht:approx_traffic>
              <ht:news_item_title>SpaceX lands rover on Mars</ht:news_item_title>
              <ht:news_item_url>https://example/mars</ht:news_item_url>
            </item>
          </channel>
        </rss>
        """
        return httpx.Response(200, text=rss)

    respx.get(server.GTRENDS_RSS_URL).mock(side_effect=_respond)

    r = await server.trends()
    assert captured["geo"] == "US"  # GeoIP-derived
    assert r["country_code"] == "US"
    assert r["items"][0]["query"] == "Mars mission"
    assert r["items"][0]["approx_traffic"] == "500,000+"
    assert r["items"][0]["related_news_title"] == "SpaceX lands rover on Mars"


@respx.mock
async def test_trends_explicit_country_code_skips_geoip():
    # No GeoIP mock — if the tool reaches for GeoIP it'll 404.
    captured: dict[str, str] = {}

    def _respond(req):
        captured.update(dict(req.url.params))
        return httpx.Response(
            200,
            text='<rss><channel><item><title>t1</title></item></channel></rss>',
        )

    respx.get(server.GTRENDS_RSS_URL).mock(side_effect=_respond)

    r = await server.trends(country_code="DE")
    assert captured["geo"] == "DE"
    assert r["country_code"] == "DE"


# ── Domain filter (MCP_ENABLED_DOMAINS) ─────────────────────────────────


def test_enabled_domains_empty_keeps_all_visible(monkeypatch):
    """Unset / empty-string → no filtering. Covers the default case so
    a regression in the parsing can't silently shrink the catalog."""
    import importlib
    import server as srv

    monkeypatch.delenv("MCP_ENABLED_DOMAINS", raising=False)
    srv = importlib.reload(srv)
    assert srv.ENABLED_DOMAINS == frozenset()
    # Apply-filter on a fake list should return it unchanged.
    class _T:
        def __init__(self, n): self.name = n
    fake = [_T("weather"), _T("web"), _T("get_current_weather_in_city")]
    assert srv._apply_domain_filter(fake) == fake


def test_enabled_domains_invalid_name_fails_at_boot(monkeypatch):
    import importlib
    import server as srv

    monkeypatch.setenv("MCP_ENABLED_DOMAINS", "weather,frobnicate")
    with pytest.raises(RuntimeError, match="unknown domain"):
        importlib.reload(srv)
    monkeypatch.delenv("MCP_ENABLED_DOMAINS", raising=False)
    importlib.reload(srv)  # restore


@pytest.mark.asyncio
async def test_enabled_domains_narrows_off_mode_surface(monkeypatch):
    """`MCP_ENABLED_DOMAINS=weather,knowledge` in off mode drops every
    non-weather/non-knowledge narrow from `tools/list`, keeps the
    rest of the narrow catalog intact. Location / time / web tools
    must not appear."""
    import importlib
    import server as srv

    monkeypatch.setenv("MCP_ROUTER_MODE", "off")
    monkeypatch.setenv("MCP_ENABLED_DOMAINS", "weather,knowledge")
    srv = importlib.reload(srv)
    try:
        from mcp.types import ListToolsRequest

        handler = srv.mcp._mcp_server.request_handlers[ListToolsRequest]
        visible = {t.name for t in (await handler(ListToolsRequest(method="tools/list"))).root.tools}

        # Weather + knowledge narrows present.
        assert "get_current_weather_in_city" in visible
        assert "get_wikipedia_summary" in visible
        assert "calculate" in visible
        # Web / geo / time narrows filtered out.
        assert "web_search" not in visible
        assert "news" not in visible
        assert "detect_my_location_by_ip" not in visible
        assert "get_current_date" not in visible
    finally:
        monkeypatch.delenv("MCP_ROUTER_MODE", raising=False)
        monkeypatch.delenv("MCP_ENABLED_DOMAINS", raising=False)
        importlib.reload(srv)


@pytest.mark.asyncio
async def test_enabled_domains_narrows_fat_lean_surface(monkeypatch):
    """In fat_tools_lean mode the filter picks a subset of the 5 fat
    tools. This is the primary production use case — the platform
    chat sidecar disabling `web` for deployments where the model
    shouldn't reach the general internet.
    """
    import importlib
    import server as srv

    monkeypatch.setenv("MCP_ROUTER_MODE", "fat_tools_lean")
    monkeypatch.setenv("MCP_ENABLED_DOMAINS", "weather,geo,knowledge")
    srv = importlib.reload(srv)
    try:
        from mcp.types import ListToolsRequest

        handler = srv.mcp._mcp_server.request_handlers[ListToolsRequest]
        visible = {t.name for t in (await handler(ListToolsRequest(method="tools/list"))).root.tools}
        assert visible == {"weather", "geo", "knowledge"}
        assert "web" not in visible
    finally:
        monkeypatch.delenv("MCP_ROUTER_MODE", raising=False)
        monkeypatch.delenv("MCP_ENABLED_DOMAINS", raising=False)
        importlib.reload(srv)


# ── Fat-router coverage for web domain ─────────────────────────────────


@pytest.mark.asyncio
async def test_fat_lean_web_dispatch(monkeypatch):
    """`web(action="search", params={"query": "..."})` in lean mode must
    reach the narrow `web_search` impl. Mirror test for the other
    three web actions."""
    import importlib
    import server as srv

    monkeypatch.setenv("MCP_ROUTER_MODE", "fat_tools_lean")
    monkeypatch.delenv("MCP_ENABLED_DOMAINS", raising=False)
    srv = importlib.reload(srv)
    try:
        import fat_tools_lean

        calls: list[tuple[str, tuple, dict]] = []

        async def _stub_web_search(q, limit=8):
            calls.append(("web_search", (q, limit), {}))
            return {"results": []}

        async def _stub_news(query=None, topic=None, lang=None, limit=10):
            calls.append(("news", (query, topic, lang, limit), {}))
            return {"items": []}

        async def _stub_hn(category="top", limit=15):
            calls.append(("hackernews", (category, limit), {}))
            return {"items": []}

        async def _stub_trends(country_code=None, limit=15):
            calls.append(("trends", (country_code, limit), {}))
            return {"items": []}

        monkeypatch.setattr(srv, "web_search", _stub_web_search)
        monkeypatch.setattr(srv, "news", _stub_news)
        monkeypatch.setattr(srv, "hackernews", _stub_hn)
        monkeypatch.setattr(srv, "trends", _stub_trends)

        await fat_tools_lean.web("search", {"query": "kyiv"})
        await fat_tools_lean.web("news", {"topic": "tech"})
        await fat_tools_lean.web("hackernews", {"category": "show"})
        await fat_tools_lean.web("trends", {"country_code": "UA"})

        called = [c[0] for c in calls]
        assert called == ["web_search", "news", "hackernews", "trends"]
        assert calls[0][1] == ("kyiv", 8)
        assert calls[1][1] == (None, "tech", None, 10)
        assert calls[2][1] == ("show", 15)
        assert calls[3][1] == ("UA", 15)
    finally:
        monkeypatch.delenv("MCP_ROUTER_MODE", raising=False)
        importlib.reload(srv)
