from __future__ import annotations

import collections
import functools
import hashlib
import inspect
import json
import os
import secrets
import time
from datetime import date, datetime, timezone as _tz
from typing import Literal
from zoneinfo import ZoneInfo

import httpx
from mcp.server.fastmcp import Context, FastMCP
from pydantic import Field, create_model

TRANSPORT = os.getenv("MCP_TRANSPORT", "stdio")
HOST = os.getenv("MCP_HOST", "0.0.0.0")
PORT = int(os.getenv("MCP_PORT", "8000"))
AUTH_TOKEN = os.getenv("MCP_AUTH_TOKEN", "").strip()

# Default `fat_tools_lean` since 2026-04-22 — see
# docs/tool-catalog-scaling.md for the measurements. Collapses the
# 23 narrow tools into 4 fat domain-tools with an `(action, params)`
# signature; catalog drops 5289 → 1090 tokens (−79%) with zero hit-rate
# regression on qwen3.5:9b (93.2% both ways). Set `MCP_ROUTER_MODE=off`
# to get the historical monolith surface — needed for nightly eval
# baseline continuity and as an escape hatch if any specific client
# turns out to misbehave on the fat surface.
#
# Other modes kept behind this flag: `fat_tools` (fat with named
# kwargs, +800 tokens vs lean) and `list_changed` (spec-correct
# dynamic router — confirmed unsupported by mcphost / Open WebUI,
# kept as a reference implementation only).
ROUTER_MODE = os.getenv("MCP_ROUTER_MODE", "fat_tools_lean").strip().lower()

# ── Tool-calling policy (#19) ──────────────────────────────────────────────
#
# Rendered into `InitializeResult.instructions` so every MCP client
# (mcphost, OWUI, custom) seeds the model with a short tool-policy
# preamble. Short on purpose — long instruction blocks get skipped by
# small models. The 2026-04-20 live session got stuck in an 11-call
# identical-argument loop across 37 min; this preamble + the
# server-side duplicate detector below are the two-part defence.
_INSTRUCTIONS = (
    "Tool-calling policy (read once per session):\n"
    "- Call at most 3 tools per user turn. If you still lack what you need, ASK THE USER.\n"
    "- Never repeat an identical tool call in the same turn — use the previous result or ask.\n"
    "- When a tool response has `relay_to_user: false`, you MUST clarify with the user before replying. Do not try alternate tools.\n"
    "- Short `guidance` strings in tool responses are instructions — follow them literally."
)

mcp = FastMCP(
    "weather",
    instructions=_INSTRUCTIONS,
    host=HOST, port=PORT,
    # Router mode needs a live session to push the
    # `notifications/tools/list_changed` event over the SSE stream —
    # stateless HTTP terminates the session at the end of the RPC call,
    # which would drop the notification. Everywhere else (default
    # monolith) stays stateless so the sidecar deployment topology
    # on the platform doesn't need session affinity.
    stateless_http=(ROUTER_MODE == "off"),
)


# ── Duplicate-call detector (#19) ──────────────────────────────────────────
#
# Rolling window of recent tool-call fingerprints. Per-process scope
# is equivalent to per-session state in our sidecar topology — each
# chat session runs in its own Pod with its own mcp-weather process,
# so process-local memory IS the session.
#
# Tuple is (fingerprint, monotonic_timestamp). maxlen=10 caps memory;
# entries older than `_LOOP_WINDOW_SECONDS` are pruned on each check.
_RECENT_CALLS: "collections.deque[tuple[str, float]]" = collections.deque(maxlen=10)
_LOOP_WINDOW_SECONDS = 120


def _reset_recent_calls() -> None:
    """Clear the detector's state. Exposed for tests — the fixture wipes
    between cases so a `get_current_weather_in_city("Kyiv")` in test A
    doesn't short-circuit the same call in test B."""
    _RECENT_CALLS.clear()


def _call_fingerprint(tool_name: str, arguments: dict) -> str:
    # Sort arg keys so {country: X, language: Y} vs {language: Y,
    # country: X} hash identically — exactly the argument-order flip
    # observed on turn 9 of the 2026-04-20 mcphost loop. `default=str`
    # survives non-JSON-native types (date, float with nan) without
    # raising; the fingerprint just needs to be stable within the window.
    payload = json.dumps({"t": tool_name, "a": arguments}, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _detect_and_record_call(fingerprint: str) -> bool:
    """True if this fingerprint matches a recent call still inside the window.

    Always appends the current call (even on duplicates) so the window
    stays fresh. Returning True does NOT stop the caller from running —
    the decision to short-circuit is the caller's.
    """
    now = time.monotonic()
    # Prune the left side of the deque until the head is within window.
    while _RECENT_CALLS and now - _RECENT_CALLS[0][1] > _LOOP_WINDOW_SECONDS:
        _RECENT_CALLS.popleft()
    is_dup = any(fp == fingerprint for fp, _ in _RECENT_CALLS)
    _RECENT_CALLS.append((fingerprint, now))
    return is_dup


def _loop_guarded(fn):
    """Short-circuit duplicate tool calls with `relay_to_user=false`.

    Wrapped tool, when called a second time with the same arguments
    inside the window, returns an envelope telling the LLM to use the
    previous result or clarify — never touches the upstream service
    again. Signature is preserved via functools.wraps so FastMCP's
    schema introspection still sees the original typed parameters.
    """
    sig = inspect.signature(fn)

    @functools.wraps(fn)
    async def wrapper(*args, **kwargs):
        try:
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            arguments = dict(bound.arguments)
        except TypeError:
            # Fall back to kwargs if binding fails (defensive — should
            # not happen for FastMCP-dispatched calls which always use
            # kwargs derived from the input schema).
            arguments = dict(kwargs)
        fingerprint = _call_fingerprint(fn.__name__, arguments)
        if _detect_and_record_call(fingerprint):
            return _respond(
                {
                    "tool_name": fn.__name__,
                    "arguments": arguments,
                    "duplicate_of_recent_call": True,
                },
                relay_to_user=False,
                guidance=(
                    f"Duplicate call: you already ran `{fn.__name__}` with "
                    "these arguments in the last few minutes. Use that "
                    "previous result or ask the user to clarify — do NOT retry."
                ),
            )
        return await fn(*args, **kwargs)

    return wrapper

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
# Archive-api covers 1940-present. Same query grammar as /forecast but
# requires `start_date` / `end_date` instead of `forecast_days`.
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
# Free, no-auth air-quality endpoint. Returns PM2.5 / PM10 / ozone /
# nitrogen-dioxide / sulphur-dioxide / carbon-monoxide + European and
# US AQI indices.
AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
WIKIPEDIA_SUMMARY_URL = "https://{lang}.wikipedia.org/api/rest_v1/page/summary/{title}"
RESTCOUNTRIES_ALPHA_URL = "https://restcountries.com/v3.1/alpha/{code}"
RESTCOUNTRIES_NAME_URL = "https://restcountries.com/v3.1/name/{name}"
HOLIDAYS_URL = "https://date.nager.at/api/v3/PublicHolidays/{year}/{country_code}"
CURRENCY_URL = "https://open.er-api.com/v6/latest/{base}"
# radio-browser is a volunteer pool of geographically distributed
# mirrors. Querying any one works; the official docs recommend a DNS
# SRV lookup + round-robin, but for our low-volume tool a static
# fallback chain with explicit mirrors is simpler and more predictable.
# If the first mirror is down, we try the next — covers ~95 % of
# single-node outages without retry code on the caller side.
RADIO_BROWSER_MIRRORS = (
    "https://de1.api.radio-browser.info/json",
    "https://de2.api.radio-browser.info/json",
    "https://at1.api.radio-browser.info/json",
    "https://nl1.api.radio-browser.info/json",
    "https://fi1.api.radio-browser.info/json",
)
# The radio-browser ToS asks clients to identify themselves so abusive
# integrations can be banned without collateral damage.
RADIO_BROWSER_UA = "mcp-weather-simple/0.2 (https://github.com/rromenskyi/mcp-weather-simple)"
# Nominatim (OSM, no key) is our primary address-normalisation source.
# ToS requires a descriptive User-Agent with contact info and ≤ 1 rps
# global. Photon (Komoot/OSM, no limit at our volume, returns GeoJSON
# in a different shape) is the fallback when Nominatim is empty or
# down. See `resolve_address` for the actual fallback chain.
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
PHOTON_URL = "https://photon.komoot.io/api"
OSM_UA = RADIO_BROWSER_UA  # same self-identification string — one repo, one UA


async def _fetch_json(
    url: str | list[str] | tuple[str, ...],
    *,
    service: str,
    timeout: float = 5.0,
    params: dict | None = None,
    headers: dict | None = None,
) -> dict | list:
    """GET+JSON with a tight timeout, friendly errors and mirror fallback.

    `url` is either a single URL or a sequence of equivalent mirrors —
    in the latter case the helper tries each in order and returns the
    first successful response, bubbling the last error if all fail.
    `service` is the human name used verbatim in error strings so the
    model can pass an actionable message to the user instead of a
    Python traceback.
    """
    urls = [url] if isinstance(url, str) else list(url)
    last_exc: Exception | None = None
    for candidate in urls:
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                r = await client.get(candidate, params=params, headers=headers)
                r.raise_for_status()
                return r.json()
        except (httpx.TimeoutException, httpx.HTTPStatusError, httpx.RequestError) as e:
            last_exc = e
            # Try the next mirror on transient-looking failures (timeout,
            # 5xx, connect error). For a genuine 4xx the next mirror will
            # return the same 4xx, so we still bail after exhausting the
            # list but with a clear message.
            continue
    # All mirrors failed — translate the last exception into a human message.
    tail = f" (tried {len(urls)} mirror(s))" if len(urls) > 1 else ""
    if isinstance(last_exc, httpx.TimeoutException):
        raise RuntimeError(
            f"{service} did not respond within {timeout:.0f} s{tail} — try again in a moment."
        ) from last_exc
    if isinstance(last_exc, httpx.HTTPStatusError):
        status = last_exc.response.status_code
        # Client-side errors (4xx) usually mean the caller passed a bad
        # argument — a Wikipedia title that doesn't exist, a postal code
        # the geocoder rejects, an ISO country code that's not in the
        # database. Labelling that as "service issues" points the model
        # at the wrong root cause; the real fix is to adjust inputs and
        # retry. Server-side errors (5xx) are the genuine upstream
        # problem that the model should pass on to the user.
        if 400 <= status < 500:
            raise RuntimeError(
                f"{service} rejected the request with HTTP {status}{tail} — check the "
                "tool arguments (e.g. the city/title/code may not exist upstream)."
            ) from last_exc
        raise RuntimeError(
            f"{service} returned HTTP {status}{tail} — the service may be having issues."
        ) from last_exc
    if isinstance(last_exc, httpx.RequestError):
        raise RuntimeError(
            f"{service} is unreachable ({type(last_exc).__name__}){tail}. Network or DNS failure."
        ) from last_exc
    raise RuntimeError(f"{service} is unreachable{tail}.")  # pragma: no cover
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


# Cyrillic glyphs that exist ONLY in Ukrainian (not in Russian). A single
# occurrence of any of these forces `language=uk` on the first geocoder
# attempt — "Київ" / "Львів" etc. return NO RESULTS with `language=ru`.
_UKRAINIAN_UNIQUE_CYRILLIC = frozenset("іїєґІЇЄҐ")


def _detect_query_languages(query: str) -> list[str]:
    """Return an ordered list of Open-Meteo `language` codes to try.

    Open-Meteo's geocoder is script-biased: with `language=en` it indexes
    the Latin name of every place, so a Cyrillic "Москва" returns only
    Tajik villages named "Moskva" instead of Moscow-RU. Passing the
    right `language=` surfaces the native-script index.

    A *list* rather than a single code because some scripts are
    ambiguous across languages that Open-Meteo indexes separately:
      - Cyrillic shared by Russian / Ukrainian / Bulgarian / Serbian:
        "Одеса" (Ukrainian) has no `і ї є ґ`, so we fall back from
        `ru` → `uk` on empty results. "Київ" / "Львів" contain
        Ukrainian-unique glyphs and go straight to `uk` first.
      - CJK Han ideographs shared by Chinese / Japanese / Korean:
        "横浜" returns only Chinese mis-matches under `zh`, but
        resolves to Yokohama-JP under `ja`. Order `zh` → `ja` → `ko`.
      - Everything else is unambiguous (kana → ja, Hangul → ko,
        Greek → el, Arabic → ar, Hebrew → he) and needs a single-item
        list; Latin queries fall through to `en`.

    `_geocode` and `search_places` call this helper and iterate through
    the returned list, stopping at the first language that yields a
    non-empty result set.
    """
    # Unambiguous script hits short-circuit: any Kana or Hangul glyph
    # fully specifies the language, even when the query also contains
    # shared Han ideographs ("東京タワー" → ja, "서울특별시" → ko).
    for ch in query:
        code = ord(ch)
        if 0x3040 <= code <= 0x30FF:
            return ["ja"]  # Hiragana + Katakana
        if 0xAC00 <= code <= 0xD7AF:
            return ["ko"]  # Hangul syllables
    # Ukrainian-specific Cyrillic glyphs take priority over generic
    # Cyrillic so "Київ" goes straight to `uk` and never wastes a round
    # trip on `ru` (which returns zero results).
    if any(ch in _UKRAINIAN_UNIQUE_CYRILLIC for ch in query):
        return ["uk", "ru"]
    for ch in query:
        code = ord(ch)
        if 0x0400 <= code <= 0x052F:
            # Generic Cyrillic — Russian index is largest, Ukrainian is a
            # real fallback because city names like "Одеса" contain only
            # shared glyphs and return empty under `ru`.
            return ["ru", "uk"]
        if 0x0370 <= code <= 0x03FF:
            return ["el"]  # Greek
        if 0x0600 <= code <= 0x06FF:
            return ["ar"]  # Arabic
        if 0x0590 <= code <= 0x05FF:
            return ["he"]  # Hebrew
        if 0x4E00 <= code <= 0x9FFF:
            # Han ideographs are shared across zh / ja / ko. `zh` first
            # (largest index), then `ja`, then `ko` — each fallback
            # catches a script-sharing city name the previous one missed.
            return ["zh", "ja", "ko"]
    return ["en"]


# Kept as a thin alias for callers that only need the primary language
# (e.g. tests or logging) — equivalent to `_detect_query_languages(q)[0]`.
def _detect_query_language(query: str) -> str:
    return _detect_query_languages(query)[0]


def _city_not_found_error(query: str, country_code: str | None) -> str:
    # Open-Meteo matches the whole `name` string literally, so
    # "Bountiful, Utah, 84010" returns nothing even though Bountiful-UT
    # is a perfect match for the first token. Rather than leaving the
    # model guessing, we detect a comma in an empty-result query and
    # hand back a self-correcting hint: split on the first comma, keep
    # the head as the probable place name, and suggest re-calling with
    # `country_code` set.
    if "," in query:
        head = query.split(",", 1)[0].strip()
        suggestion = f"try {head!r}"
        if not country_code:
            suggestion += ' with country_code="US" (ISO-3166 alpha-2)'
        return (
            f"City not found: {query!r}. This endpoint matches the whole "
            f"string literally — pass a SINGLE token (place name OR postal "
            f"code, never a comma-separated address). Put country into "
            f"`country_code` (ISO-3166 alpha-2). {suggestion}."
        )
    tail = f" in {country_code}" if country_code else ""
    return f"City not found: {query}{tail}"


# Populated-place GeoNames codes; used by the ambiguity detector to
# decide whether two candidates are "homonyms that would fool a user"
# (both towns/villages/cities) vs "same name different kinds of place"
# (city + mountain + lake — the weather tool can still pick top-1
# because the user almost certainly meant the populated one).
_POPULATED_PLACE_CODES = frozenset({
    "PPL", "PPLA", "PPLA2", "PPLA3", "PPLA4", "PPLA5", "PPLC",
    "PPLL", "PPLS",
})


def _detect_ambiguity(
    candidates: list[dict],
    country_code: str | None,
    query: str,
) -> str | None:
    """Decide whether _geocode's hits are ambiguous enough to short-circuit.

    Returns a short reason string when we should ask the user to clarify,
    or None when picking `candidates[0]` is safe.

    The primary gate is the **10× population ratio** among the top-2
    populated candidates (ROADMAP rule 2). This is the key insight:
    Moscow-RU (12.5 M) vs Moscow-ID (25 k) = 500× — top-1 is the
    obvious intent and silent commit is fine. Kyiv (3 M) vs a Tajik
    hamlet (1.3 k) = 2300× — same, keep silent. Springfield-IL
    (114 k) vs Springfield-MA (155 k) = 1.4× — genuinely ambiguous,
    ask. The same gate catches intra-country homonyms ("Bountifuls
    in UT and CO") so country_code does NOT have to be unset.

    Cross-country without population data (pop fields missing from
    both candidates) falls back to "cross-country, can't verify" —
    the safety net exists because some Open-Meteo responses omit
    `population`.

    Postal codes bypass the population gate: "10001" matching NYC-US
    and Troyes-FR legitimately needs disambiguation, population
    ratio is not meaningful for "which country's postal code".
    """
    if len(candidates) < 2:
        return None
    populated = [c for c in candidates if c.get("feature_code") in _POPULATED_PLACE_CODES]
    if len(populated) < 2:
        return None

    top, second = populated[0], populated[1]
    cc_top = (top.get("country_code") or "").upper()
    cc_second = (second.get("country_code") or "").upper()
    is_cross_country = bool(cc_top and cc_second and cc_top != cc_second)
    pop_top = top.get("population") or 0
    pop_second = second.get("population") or 0

    # Postal-code rule: numeric query with cross-country matches fires
    # regardless of population, because postal codes are namespaced
    # per-country and a cross-country spread is the "10001 NYC / Troyes"
    # collision. Matching digits is a pragmatic proxy for "looks like
    # a postal code" — the geocoder wouldn't surface non-matching
    # candidates at this point anyway.
    if is_cross_country and query.replace(" ", "").replace("-", "").isdigit():
        return f"postal code matches multiple countries ({sorted({cc_top, cc_second})})"

    # Population-ratio rule: if both populations are known AND the gap
    # is under 10×, top-1's "obvious" win is unsafe.
    if pop_top > 0 and pop_second > 0 and pop_top / pop_second < 10.0:
        if is_cross_country:
            return f"top two places have comparable populations across countries ({cc_top} and {cc_second})"
        return f"top two places have comparable populations ({pop_top:,} vs {pop_second:,})"

    # Fallback: cross-country with missing population data. Being
    # conservative — if the geocoder omits populations for the top-2
    # we can't do the ratio check, so err toward asking when there's
    # any cross-country signal.
    if is_cross_country and (pop_top == 0 or pop_second == 0):
        return f"populated places across countries ({cc_top} and {cc_second}) with missing population data"

    return None


def _candidate_label(c: dict) -> str:
    """Render a geocoder candidate as a human-readable disambiguation
    label: `"<name>, <admin1>, <country_code>"` with missing parts
    dropped. Used by both the `relay_to_user=false` envelope AND by
    the MCP elicitation schema so the user sees identical wording
    regardless of which disambiguation path fires.
    """
    parts = [c.get("name") or "?"]
    if c.get("admin1"):
        parts.append(c["admin1"])
    if c.get("country_code"):
        parts.append(c["country_code"])
    return ", ".join(parts)


async def _try_elicit_disambiguation(query: str, candidates: list[dict]) -> dict | None:
    """Ask the MCP client to elicit a choice from the user (#17 via spec 2025-11-25).

    Spec path: `elicitation/create` with form mode, enum-constrained
    `choice` field. Forward-compatible with Claude Desktop + any
    FastMCP-based host that advertises `capabilities.elicitation`.

    Returns the chosen candidate dict on success, or None if:
      * there's no active request context (running outside a tool call,
        e.g. unit tests without an ASGI session),
      * the client didn't advertise the elicitation capability (FastMCP
        raises internally when the session can't route the request),
      * the user declined / cancelled,
      * label→candidate reverse lookup failed (shouldn't happen given
        the schema constrains `choice` to our own labels).

    Callers fall back to the `_ambiguity_response` envelope on None, so
    clients without elicitation support still get the `relay_to_user:
    false` signal and the LLM can do the clarification itself.
    """
    try:
        ctx = mcp.get_context()
        # `ctx.request_context` is a @property that raises ValueError
        # when we're running outside a request (unit tests, stdio
        # startup, etc.), so we probe via the private attribute
        # instead of relying on truthiness.
        if ctx is None or getattr(ctx, "_request_context", None) is None:
            return None
    except Exception:
        return None

    top = candidates[:5]
    labels = [_candidate_label(c) for c in top]
    by_label = dict(zip(labels, top))

    # Dynamic enum via Literal[tuple(...)] — Pydantic turns this into a
    # JSON-Schema `enum` which FastMCP/MCP forwards as the spec-defined
    # `requestedSchema` for form-mode elicitation.
    choice_type = Literal[tuple(labels)]
    ChoiceModel = create_model(
        "_DisambigChoice",
        choice=(choice_type, Field(..., description="Which place did the user mean?")),
    )

    try:
        result = await ctx.elicit(
            message=f"Which '{query}' did you mean?",
            schema=ChoiceModel,
        )
    except Exception:
        return None

    if result.action != "accept" or result.data is None:
        return None
    return by_label.get(getattr(result.data, "choice", None))


def _ambiguity_response(query: str, candidates: list[dict], reason: str, country_code: str | None) -> dict:
    """Build the short-circuit envelope for an ambiguous _geocode result.

    `relay_to_user=False` is the contract: the LLM MUST clarify before
    answering. Guidance names the first few candidates so the model
    can echo them back to the user without a follow-up `search_places`
    call.

    This is the FALLBACK path: `_try_elicit_disambiguation` runs first
    and returns a resolved hit when the client supports elicitation.
    """
    # Limit to 5 so guidance stays short even for "Springfield" (36 hits).
    top = candidates[:5]
    names = "; ".join(_candidate_label(c) for c in top)
    # Narrowing advice: if no country_code yet, the easy rewrite is to
    # add one. If country_code is already set (intra-country homonyms
    # like the 5 Bountifuls), the weather tools don't accept `admin1`
    # today — the escape hatch is `get_weather_by_coordinates` which
    # takes lat/lon straight from the chosen candidate.
    if country_code:
        narrow = (
            "Take the chosen candidate's `latitude` and `longitude` "
            "and call `get_weather_by_coordinates` directly."
        )
    else:
        narrow = (
            "Then re-call with `country_code` set (ISO-3166 alpha-2) "
            "to narrow the geocoder."
        )
    guidance = f"Ambiguous: {reason}. Ask the user to pick one: {names}. {narrow}"
    return _respond(
        {
            "query": query,
            "candidates": top,
            "ambiguity_reason": reason,
        },
        relay_to_user=False,
        guidance=guidance,
    )


async def _geocode(city: str, country_code: str | None = None) -> dict:
    # Pull candidates so we can filter by `country_code` client-side —
    # Open-Meteo's geocoding endpoint does not accept a country filter and
    # returns towns mixed with mountains, lakes, neighborhoods and islands
    # bearing the same name. We intentionally do NOT silently exclude
    # non-city hits here — a caller that wants only populated places can
    # use `search_places(feature_types=["city"])` and pick explicitly, while
    # someone asking about weather on Mt. Everest or in the Bountiful
    # Islands should still get a useful top hit back.
    #
    # `count=10` is enough when the caller did not ask to disambiguate by
    # country — Open-Meteo ranks by relevance/population and the user
    # gets "the" match. With an explicit `country_code` filter we pull
    # the API maximum (100) because the desired country may sit far
    # below the top 10 for common homonyms ("London" → UK dominates, so
    # London-CA / London-US sit outside the first page and a client-side
    # filter on count=10 would wrongly raise "City not found").
    #
    # Iterate through every candidate language for the query's script
    # until we find one that returns something. Most queries hit on the
    # first attempt; shared-script queries ("Одеса", "横浜") gracefully
    # fall back to the next language without a behaviour change for the
    # caller.
    geocode_count = 100 if country_code else 10
    results: list[dict] = []
    for lang in _detect_query_languages(city):
        data = await _fetch_json(
            GEOCODE_URL,
            service="Open-Meteo geocoder",
            params={"name": city, "count": geocode_count, "language": lang},
        )
        results = data.get("results") or []
        if results:
            break
    if country_code:
        cc_upper = country_code.strip().upper()
        filtered = [h for h in results if (h.get("country_code") or "").upper() == cc_upper]
        if filtered:
            results = filtered
    if not results:
        raise ValueError(_city_not_found_error(city, country_code))
    # Ambiguity detection uses the RAW hits (not _annotate'd yet) so the
    # detector can read `feature_code` / `population` directly. When a
    # rule fires we bail with a tagged dict that tool wrappers recognise
    # — see `_resolve_place` for the unpacking pattern.
    reason = _detect_ambiguity(results[:5], country_code, query=city)
    if reason:
        annotated = [_annotate(h) for h in results[:5]]
        return {
            "_ambiguous": True,
            "_ambiguity_reason": reason,
            "_candidates": annotated,
        }
    # Internal helper — returns a bare annotated dict, callers wrap it
    # with _respond() when they're exposed as a tool. _geocode itself
    # is not a @mcp.tool.
    return _annotate(results[0])


async def _resolve_place(
    city: str,
    country_code: str | None = None,
) -> tuple[dict | None, dict | None]:
    """Geocode and short-circuit on ambiguity.

    Returns `(hit, None)` on success or `(None, envelope)` when the
    result was ambiguous. Every @mcp.tool that needs a single location
    uses this pattern:

        hit, clarify = await _resolve_place(city, country_code)
        if clarify:
            return clarify
        # ... use hit["latitude"] etc.

    This keeps each tool's happy path unchanged while making the
    short-circuit one line of boilerplate instead of every tool
    reinventing the `relay_to_user=False` response.
    """
    loc = await _geocode(city, country_code=country_code)
    if loc.get("_ambiguous"):
        # Forward-compatible path per MCP spec 2025-11-25: ask the client
        # to elicit a choice from the user. On success the caller gets a
        # regular hit as if the geocoder had returned top-1. Clients
        # that didn't advertise the elicitation capability (OWUI,
        # mcphost today) silently fall through to the envelope below.
        chosen = await _try_elicit_disambiguation(city, loc["_candidates"])
        if chosen is not None:
            return chosen, None
        return None, _ambiguity_response(
            query=city,
            candidates=loc["_candidates"],
            reason=loc["_ambiguity_reason"],
            country_code=country_code,
        )
    return loc, None


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
@_loop_guarded
async def find_place_coordinates(city: str, country_code: str | None = None) -> dict:
    """Resolve a city name or postal code to lat/lon, country and timezone.

    `city` MUST be a single token — `"Kyiv"`, `"Paris"`, `"Bountiful"`
    or a postal code `"90210"`, `"84010"`. NEVER a comma-separated
    address: `"Paris, France"`, `"Bountiful, Utah, 84010"` and
    `"Kyiv, Ukraine"` all return empty — country goes into
    `country_code` below, not the name. For a full address use
    `resolve_address` instead.

    `country_code` is an optional ISO-3166 alpha-2 hint ("US", "UA")
    to disambiguate homonyms like "Moscow, RU" vs "Moscow, ID".

    When multiple comparable places match (Springfield, the 5
    Bountifuls, …) the response flips `relay_to_user` to `false` and
    lists candidates — the LLM must ask the user which one before
    re-calling.
    """
    hit, clarify = await _resolve_place(city, country_code=country_code)
    if clarify:
        return clarify
    return _respond(hit)


@mcp.tool()
@_loop_guarded
async def search_places(
    query: str,
    country_code: str | None = None,
    feature_types: list[str] | None = None,
    limit: int = 5,
) -> dict:
    """List candidates for an ambiguous place query: town, mountain,
    lake, park, airport, etc. Each candidate carries a `feature_type`
    label so the caller can match user intent (weather on Mt. Bountiful
    vs weather in Bountiful are different places).

    - `query`: single token — place name or postal code. NEVER a
      comma-separated address; put country into `country_code`. For
      a full postal address use `resolve_address`.
    - `country_code`: ISO-3166 alpha-2 hint ("US", "UA").
    - `feature_types`: allowlist like `["city", "village"]`
      (populated places only) or `["mountain", "peak"]`. Empty/None
      keeps every type.
    - `limit`: 1-10 (default 5).

    Use when the name is ambiguous ("Springfield"), intent is unclear
    (town vs mountain), or top-1's `feature_type` doesn't match what
    the user wants. Empty result with a comma in the query grows a
    `hint` field suggesting a rewrite.
    """
    limit = max(1, min(int(limit), 10))
    # Same fallback-chain as _geocode — try every candidate language
    # until one yields results. Important for shared-script inputs
    # (Cyrillic "Одеса", Han "横浜") that have ambiguous primary language.
    # Bump `count` to the API maximum (100) when the caller narrows by
    # `country_code` or `feature_types` — the target entries may be
    # outside the default top-10 for common homonyms (London-UK drowns
    # London-CA in the global ranking), and a too-small page size would
    # produce empty filtered results that look like false negatives.
    geocode_count = 100 if (country_code or feature_types) else 10
    results: list[dict] = []
    for lang in _detect_query_languages(query):
        data = await _fetch_json(
            GEOCODE_URL,
            service="Open-Meteo geocoder",
            params={"name": query, "count": geocode_count, "language": lang},
        )
        results = data.get("results") or []
        if results:
            break
    if country_code:
        cc_upper = country_code.strip().upper()
        results = [h for h in results if (h.get("country_code") or "").upper() == cc_upper]
    if feature_types:
        allowed = {ft.strip().lower() for ft in feature_types if ft}
        results = [h for h in results if (_feature_type(h.get("feature_code")) or "").lower() in allowed]
    candidates = [_annotate(h) for h in results[:limit]]
    response = {
        "query": query,
        "country_code": country_code,
        "feature_types": feature_types,
        "candidates": candidates,
        "ambiguous": len(candidates) > 1,
    }
    # When the query has a comma and we found nothing, surface the same
    # self-correcting hint as `_geocode` so the caller can rewrite the
    # call instead of reporting "no such place".
    if not candidates and "," in query:
        response["hint"] = _city_not_found_error(query, country_code)
    # Envelope guidance:
    # - 0 candidates → relay "nothing found" (+ hint when present).
    # - 1 candidate → relay directly.
    # - 2+ candidates → tell the model to list them if the user asked
    #   for a list, or to clarify otherwise. The `relay_to_user: true`
    #   stays on here because search_places is EXPLICITLY called for
    #   disambiguation — flipping it to false would make the model
    #   stall on every intentional "list all Springfields" request.
    #   #17 takes over the relay_to_user=false path for the implicit
    #   disambiguation case inside `_geocode`.
    if not candidates:
        guidance = response.get("hint") or f"No places found for {query!r}. Ask the user to rephrase."
    elif len(candidates) == 1:
        guidance = _GUIDANCE_DIRECT
    else:
        guidance = (
            f"{len(candidates)} candidates — if the user asked for a list, "
            "relay them; otherwise ask which one."
        )
    return _respond(response, guidance=guidance)


# ── Address normalisation (resolve_address, #16) ──────────────────────────
#
# Separate path from `_geocode` / `search_places`: those take a SINGLE
# canonical token (place name or postal code) — exactly what #14's
# docstring-first contract enforces. `resolve_address` is the ESCAPE
# HATCH for the opposite situation: the user really did type a full
# postal address (`"Bountiful, Utah, 84010"`, `"221B Baker Street,
# London"`), and the LLM can feed the free-form string here instead of
# trying to parse it.
#
# Sources (no-key, fits our usual pattern):
#   1. Nominatim (OSM, 1 rps global). Accept-language=en requests an
#      English display_name so downstream weather tools get Latin
#      input — OSM has multilingual name tags, so the SEARCH works
#      for any script regardless of header.
#   2. Photon (Komoot/OSM, no rate-limit at our volume). Returns
#      GeoJSON in a different shape; normalised in `_normalise_photon`.
#   3. Last-ditch: Nominatim again, this time with an accept-language
#      matching the input script (via `_detect_query_languages`). Helps
#      when OSM simply lacks a Latin tag for a native-script place.
# Each hop short-circuits on the first populated result.


def _normalise_nominatim_hit(hit: dict, *, address_input: str) -> dict:
    """Flatten a Nominatim `jsonv2` result into our shared output shape.

    Nominatim's `addressdetails=1` mode surfaces a nested `address`
    dict whose keys vary by hit type (`city` vs `town` vs `village`
    vs `hamlet` vs `suburb`). We pick the most-populated bucket in
    that order — matches how a human would read the address.
    """
    addr = hit.get("address") or {}
    # Pick the finest-resolution populated-place label that's present;
    # `neighbourhood` / `suburb` are intentionally NOT here — a weather
    # query for "Brooklyn" wants NYC, not the neighbourhood label.
    city = (
        addr.get("city")
        or addr.get("town")
        or addr.get("village")
        or addr.get("hamlet")
        or addr.get("municipality")
    )
    return {
        "address": address_input,
        "city": city,
        "state": addr.get("state") or addr.get("region"),
        "country": addr.get("country"),
        "country_code": (addr.get("country_code") or "").upper() or None,
        "postcode": addr.get("postcode"),
        "latitude": float(hit["lat"]) if hit.get("lat") is not None else None,
        "longitude": float(hit["lon"]) if hit.get("lon") is not None else None,
        "display_name": hit.get("display_name"),
        "source": "nominatim",
    }


def _normalise_photon_feature(feat: dict, *, address_input: str) -> dict:
    """Flatten a Photon GeoJSON feature into our shared output shape.

    Photon returns `properties` with a flat set of optional keys and
    `geometry.coordinates` in [lon, lat] order (GeoJSON standard).
    Falls back to `name` when no `city`/`locality` is present —
    useful for island / rural matches where Photon skips city tags.
    """
    props = feat.get("properties") or {}
    geom = feat.get("geometry") or {}
    coords = geom.get("coordinates") or [None, None]
    lon, lat = (coords + [None, None])[:2]
    city = (
        props.get("city")
        or props.get("locality")
        or props.get("town")
        or props.get("village")
        or props.get("name")
    )
    cc = (props.get("countrycode") or "").upper() or None
    # Photon's `name` can already be the best human label — use it to
    # synthesise a display_name when the feature doesn't ship one.
    display_parts = [p for p in (props.get("name"), props.get("city"),
                                  props.get("state"), props.get("country"),
                                  props.get("postcode")) if p]
    return {
        "address": address_input,
        "city": city,
        "state": props.get("state"),
        "country": props.get("country"),
        "country_code": cc,
        "postcode": props.get("postcode"),
        "latitude": float(lat) if lat is not None else None,
        "longitude": float(lon) if lon is not None else None,
        "display_name": ", ".join(display_parts) if display_parts else props.get("name"),
        "source": "photon",
    }


async def _try_nominatim(address: str, country_code: str | None, accept_language: str) -> dict | None:
    params: dict[str, str | int] = {
        "q": address,
        "format": "jsonv2",
        "limit": 1,
        "addressdetails": 1,
    }
    if country_code:
        params["countrycodes"] = country_code.strip().lower()
    try:
        data = await _fetch_json(
            NOMINATIM_URL,
            service="Nominatim",
            timeout=5.0,
            params=params,
            headers={"User-Agent": OSM_UA, "Accept-Language": accept_language},
        )
    except Exception:
        return None
    if not isinstance(data, list) or not data:
        return None
    return _normalise_nominatim_hit(data[0], address_input=address)


async def _try_photon(address: str, country_code: str | None) -> dict | None:
    params: dict[str, str | int] = {"q": address, "limit": 1}
    if country_code:
        # Photon uses lowercase ISO-2 via `lang` param? No — it uses
        # `layer` and doesn't have a clean country filter. Skip.
        pass
    try:
        data = await _fetch_json(
            PHOTON_URL,
            service="Photon",
            timeout=5.0,
            params=params,
            headers={"User-Agent": OSM_UA},
        )
    except Exception:
        return None
    features = data.get("features") if isinstance(data, dict) else None
    if not features:
        return None
    feat = features[0]
    # Apply country_code filter client-side if requested. Photon puts
    # the ISO-2 in `properties.countrycode`; a mismatch means we asked
    # for US and got a Canadian hit on a spurious lexical match.
    if country_code:
        cc_want = country_code.strip().upper()
        cc_got = ((feat.get("properties") or {}).get("countrycode") or "").upper()
        if cc_got and cc_got != cc_want:
            return None
    return _normalise_photon_feature(feat, address_input=address)


@mcp.tool()
@_loop_guarded
async def resolve_address(address: str, country_code: str | None = None) -> dict:
    """Parse a free-form postal address into structured components.

    Use when the input is a full address — multiple comma-separated
    parts, street number, or a postcode attached: `"Bountiful, Utah,
    84010"`, `"221B Baker Street, London"`, `"Бульвар Лобановського,
    23, Київ"`. For a plain city name or bare postcode use
    `find_place_coordinates` or `get_current_weather_in_city` directly.

    Returns latitude, longitude, and English-normalised `city` +
    `country_code` ready to feed into the weather / time tools.
    For current weather on the resolved point, pass latitude/longitude
    into `get_weather_by_coordinates` as the one-step shortcut.

    `country_code` (ISO-3166 alpha-2, optional) narrows cross-border
    ambiguity.
    """
    if not address or not address.strip():
        raise ValueError("address must be a non-empty string")
    query = address.strip()

    # Hop 1: Nominatim, English output. OSM's multilingual name tags
    # let the search match any script; accept-language=en picks the
    # Latin display where available, keeping downstream weather tools
    # on familiar ground.
    hit = await _try_nominatim(query, country_code, accept_language="en")
    if hit:
        return _respond(hit)

    # Hop 2: Photon. GeoJSON format, normalised to the same shape.
    hit = await _try_photon(query, country_code)
    if hit:
        return _respond(hit)

    # Hop 3: Nominatim with script-matched accept-language. Last
    # chance when OSM has only a native-script name tag for the place
    # and the Latin form we requested in hop 1 wasn't present.
    for lang in _detect_query_languages(query):
        if lang == "en":
            continue  # already tried in hop 1
        hit = await _try_nominatim(query, country_code, accept_language=lang)
        if hit:
            return _respond(hit)

    return _respond(
        {"address": query, "country_code": country_code, "candidates": []},
        relay_to_user=False,
        guidance=(
            f"No address resolved for {query!r}. Ask the user to "
            "rephrase or provide more detail (city, country)."
        ),
    )


# ── No-arg "user-question" shortcuts ───────────────────────────────────────
#
# Pre-composed tools whose NAME alone tells a weak model which user question
# they answer. Each one auto-detects the caller's city via GeoIP (so the
# user doesn't have to name their city) and then calls the appropriate
# lower-level tool. The lower-level tools remain available for precise
# follow-ups — these are just the obvious defaults.


async def _here_city() -> tuple[str, str | None]:
    """Resolve the caller's city + country_code from its public IP.

    Uses `_detect_my_location_by_ip_impl` rather than the @mcp.tool
    wrapper so the GeoIP call does NOT register with the #19
    duplicate-call detector — only USER-facing tool invocations
    should populate that window.
    """
    body, _ = await _detect_my_location_by_ip_impl()
    city = body.get("city")
    cc = body.get("country_code")
    if not city:
        raise ValueError(
            "GeoIP lookup did not resolve a city for the caller's public IP. "
            "Ask the user to name a city and call the explicit tool instead."
        )
    return city, cc


# ── Response envelope (#18) ────────────────────────────────────────────────
#
# Every successful tool response carries two top-level fields:
#   relay_to_user: bool - true  = model can answer directly from this data
#                         false = model MUST clarify with the user first
#                                 (e.g. ambiguous input, multiple candidates,
#                                 duplicate call — see #17, #19)
#   guidance:      str  - plain-English one-liner telling the model what to
#                         do with the body. Small models follow instructions
#                         better than they interpret a controlled vocabulary
#                         like `confidence: enum`, so we use prose.
#
# The envelope is the substrate for the loop-breaking work: #17 flips
# `relay_to_user` to false when the geocoder finds multiple strong
# matches, and #19's duplicate-call detector will return a short-circuit
# envelope with guidance like "you just ran this, ask the user instead".


_GUIDANCE_DIRECT = "Relay directly."

# Guidance for every GeoIP-backed path. Kept short on purpose — small
# models truncate or ignore long instructions. The full rationale
# (VPN / NAT / cluster egress specifics) is in the repo; the LLM just
# needs the action verb + the uncertainty flag.
_GUIDANCE_GEOIP_AUTODETECT = (
    "Relay with a caveat: city was auto-detected from the caller's IP and "
    "may be wrong (VPN / NAT). If the user disagrees, ask for the city."
)
_GUIDANCE_GEOIP_EXPLICIT = (
    "Relay with a caveat: city is a GeoIP approximation, may be off for "
    "VPN / mobile / data-center IPs."
)


def _respond(
    data: dict,
    *,
    relay_to_user: bool = True,
    guidance: str = _GUIDANCE_DIRECT,
) -> dict:
    # Merge the envelope on top of the payload so it always takes
    # precedence — tools that accidentally emit a `relay_to_user` or
    # `guidance` key in their body cannot override the contract.
    return {**data, "relay_to_user": relay_to_user, "guidance": guidance}


@mcp.tool()
@_loop_guarded
async def get_weather_outside_right_now() -> dict:
    """What's the weather outside right now, where the user is.

    Zero-argument shortcut for "какая погода на улице?" / "what's the
    weather outside?". Auto-detects the user's city via GeoIP and
    returns the current conditions — temperature, humidity, wind,
    precipitation. If the user is asking about a specific city by
    name, call `get_current_weather_in_city(city)` instead.
    """
    city, cc = await _here_city()
    body = await _get_current_weather_in_city_impl(city, country_code=cc)
    # On geocoder ambiguity (#17) the impl returns an envelope with
    # relay_to_user=False; pass it through verbatim instead of wrapping
    # it in our GeoIP guidance (which would override the contract).
    if body.get("relay_to_user") is False:
        return body
    return _respond(
        {**body, "location_source": "geoip_autodetected"},
        guidance=_GUIDANCE_GEOIP_AUTODETECT,
    )


@mcp.tool()
@_loop_guarded
async def get_weather_forecast_for_today() -> dict:
    """Today's forecast (high / low / conditions / rain chance) where the user is.

    Zero-argument shortcut for "какая погода сегодня?" / "what's the
    weather today?". Auto-detects the user's city via GeoIP and
    returns ONE entry from the daily forecast labelled "today".
    """
    city, cc = await _here_city()
    body = await _get_weather_forecast_impl(city, days=1, country_code=cc)
    if body.get("relay_to_user") is False:
        return body
    return _respond(
        {**body, "location_source": "geoip_autodetected"},
        guidance=_GUIDANCE_GEOIP_AUTODETECT,
    )


@mcp.tool()
@_loop_guarded
async def get_current_time_where_i_am() -> dict:
    """What time / date / timezone it is right now, where the user is.

    Zero-argument shortcut for "который сейчас час?" / "what time
    is it?" / "what's today's date?". Auto-detects the user's city
    and timezone via GeoIP. Returns local time, local date, weekday,
    UTC offset and the IANA timezone name. If the user is asking
    about a specific city instead of themselves, call
    `get_current_time_in_city(city)`.
    """
    loc, _ = await _detect_my_location_by_ip_impl()
    return _respond(
        {
            "city": loc.get("city"),
            "country": loc.get("country"),
            "country_code": loc.get("country_code"),
            "timezone_id": loc.get("timezone_id"),
            "local_date": loc.get("local_date"),
            "local_time": loc.get("local_time"),
            "weekday": loc.get("weekday"),
            "utc_offset": loc.get("utc_offset"),
            "iso_datetime": loc.get("iso_datetime"),
            "location_source": "geoip_autodetected",
        },
        guidance=_GUIDANCE_GEOIP_AUTODETECT,
    )


@mcp.tool()
@_loop_guarded
async def get_weather_forecast_for_tomorrow() -> dict:
    """Tomorrow's forecast (high / low / conditions / rain chance) where the user is.

    Zero-argument shortcut for "какая погода завтра?" / "what's the
    forecast for tomorrow?". Auto-detects the user's city via GeoIP
    and returns TWO entries from the daily forecast (today + tomorrow,
    so the model can compare); `day_label` on each says which is which.
    """
    city, cc = await _here_city()
    body = await _get_weather_forecast_impl(city, days=2, country_code=cc)
    if body.get("relay_to_user") is False:
        return body
    return _respond(
        {**body, "location_source": "geoip_autodetected"},
        guidance=_GUIDANCE_GEOIP_AUTODETECT,
    )


async def _detect_my_location_by_ip_impl(ip: str | None = None) -> tuple[dict, str]:
    """Raw GeoIP lookup — returns (body, guidance) without the envelope.

    Decoupled from `@mcp.tool` so internal callers (`_here_city`,
    shortcut tools) can invoke it without going through the #19 loop
    guard — otherwise nested calls from a shortcut fingerprint the
    GeoIP lookup AND the shortcut, turning a second shortcut invocation
    into a false-positive "duplicate" short-circuit of the inner
    GeoIP.

    Returning a 2-tuple instead of a ready envelope lets the caller
    decide whether to expose GeoIP guidance or subsume it into its
    own guidance (shortcuts do the latter).
    """
    target = f"{GEOIP_URL}/{ip}" if ip else f"{GEOIP_URL}/"
    data = await _fetch_json(target, service="ipwho.is GeoIP", timeout=5.0)
    if not data.get("success", True):
        raise ValueError(f"GeoIP lookup failed: {data.get('message', 'unknown reason')}")
    tz_id = (data.get("timezone") or {}).get("id") or "UTC"
    try:
        tz = ZoneInfo(tz_id)
    except Exception:
        tz = _tz.utc
        tz_id = "UTC"
    now = datetime.now(tz)
    body = {
        "ip": data.get("ip"),
        "city": data.get("city"),
        "region": data.get("region"),
        "country": data.get("country"),
        "country_code": data.get("country_code"),
        "latitude": data.get("latitude"),
        "longitude": data.get("longitude"),
        "timezone_id": tz_id,
        "local_time": now.strftime("%H:%M:%S"),
        "local_date": now.date().isoformat(),
        "weekday": now.strftime("%A"),
        "iso_datetime": now.isoformat(timespec="seconds"),
        "utc_offset": now.strftime("%z"),
        "location_source": "geoip_explicit" if ip else "geoip_autodetected",
    }
    guidance = _GUIDANCE_GEOIP_EXPLICIT if ip else _GUIDANCE_GEOIP_AUTODETECT
    return body, guidance


@mcp.tool()
@_loop_guarded
async def detect_my_location_by_ip() -> dict:
    """Auto-detect the CALLER'S location — takes no arguments.

    Answers "where am I?", "what's the weather here?", "what time is
    it here?" — questions where the user is asking about themselves.
    Auto-detects the caller's public IP from the incoming HTTP
    request (no argument needed). Backed by ipwho.is (no key).

    Feed the returned `city` + `country_code` straight into
    `get_current_weather_in_city`, `get_weather_forecast` etc.

    For a SPECIFIC IP (someone else's, debugging a load-balancer),
    use `lookup_ip_geolocation(ip=...)` instead — that tool takes a
    required `ip` argument and exists precisely to keep the
    "MY location" semantics of this one clean.

    Limitation: inside Kubernetes / NAT / VPN the auto-detected IP
    is the cluster / NAT egress, not the end user's browser IP.
    """
    body, guidance = await _detect_my_location_by_ip_impl(ip=None)
    return _respond(body, guidance=guidance)


@mcp.tool()
@_loop_guarded
async def lookup_ip_geolocation(ip: str) -> dict:
    """Resolve a SPECIFIC IPv4 / IPv6 to city + country + coords.

    Use this tool ONLY when the user has provided an explicit IP
    address (or pastes one from elsewhere), e.g.
    "where is 8.8.8.8 located?", "what city is 198.51.100.7 in?".

    **Do NOT** use this tool to find the user's own location — that's
    what `detect_my_location_by_ip` (no args) is for. The naming
    split is deliberate: "MY" → self-lookup no args, "lookup" →
    someone else's IP with the IP required.

    `ip` is a required IPv4 or IPv6 string (e.g. "8.8.8.8",
    "2001:4860:4860::8888"). Returns the same shape as
    `detect_my_location_by_ip`: city, region, country, country_code,
    lat / lon, timezone, local clock.

    Accuracy depends on the IP-to-geo database; mobile carrier / VPN
    / data-center addresses often resolve far from the actual user.
    """
    if not ip or not ip.strip():
        raise ValueError("ip argument is required — use detect_my_location_by_ip for self-lookup")
    body, guidance = await _detect_my_location_by_ip_impl(ip=ip.strip())
    return _respond(body, guidance=guidance)


@mcp.tool()
@_loop_guarded
async def get_current_time_in_city(city: str, country_code: str | None = None) -> dict:
    """Return the current local date, time and timezone of a city or zipcode.

    Answers "what time is it in Kyiv?" without the model having to know
    the city's offset. Internally calls the same geocoder the weather
    tools use — so `country_code` is an optional ISO-3166-1 alpha-2
    hint and the response's `name` / `country` / `admin1` confirm which
    place was picked.

    `city` must be a SINGLE token — place name ("Kyiv") or postal code
    ("10001"), NEVER a comma-separated address. Put country into
    `country_code` (ISO-3166 alpha-2 like "US", "UA").
    """
    loc, clarify = await _resolve_place(city, country_code=country_code)
    if clarify:
        return clarify
    try:
        tz = ZoneInfo(loc.get("timezone") or "UTC")
    except Exception:
        tz = _tz.utc
    now = datetime.now(tz)
    return _respond({
        "location": f"{loc['name']}, {loc.get('country', '')}".rstrip(", "),
        "admin1": loc.get("admin1"),
        "country_code": loc.get("country_code"),
        "timezone_id": loc.get("timezone"),
        "date": now.date().isoformat(),
        "time": now.strftime("%H:%M:%S"),
        "weekday": now.strftime("%A"),
        "iso_datetime": now.isoformat(timespec="seconds"),
        "utc_offset": now.strftime("%z"),
    })


@mcp.tool()
@_loop_guarded
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
    return _respond({
        "date": now.date().isoformat(),
        "weekday": now.strftime("%A"),
        "iso_datetime": now.isoformat(timespec="seconds"),
        "timezone": timezone,
    })


async def _get_current_weather_in_city_impl(city: str, country_code: str | None = None) -> dict:
    """Raw current-weather fetch — returns either a bare body dict or,
    on geocoder ambiguity (#17), a pre-built envelope with
    `relay_to_user=False`. Decoupled from @mcp.tool so shortcut tools
    can call it without hitting the #19 loop guard twice.
    """
    loc, clarify = await _resolve_place(city, country_code=country_code)
    if clarify:
        return clarify  # already an envelope (relay_to_user=False)
    data = await _fetch_json(
        FORECAST_URL,
        service="Open-Meteo forecast",
        timeout=8.0,
        params={
            "latitude": loc["latitude"],
            "longitude": loc["longitude"],
            "current": "temperature_2m,relative_humidity_2m,apparent_temperature,"
                       "precipitation,weather_code,wind_speed_10m,wind_direction_10m",
            "timezone": "auto",
        },
    )
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
@_loop_guarded
async def get_current_weather_in_city(city: str, country_code: str | None = None) -> dict:
    """Get the current weather for a city, postal code, or lat/lon name.

    `city` must be a SINGLE token — place name ("Kyiv") or postal code
    ("10001"), NEVER a comma-separated address ("Kyiv, Ukraine" fails;
    "Bountiful, Utah, 84010" fails). Put country into `country_code`
    (ISO-3166 alpha-2 like "US", "UA").

    `country_code` is an optional ISO-3166-1 alpha-2 hint to disambiguate
    homonyms (e.g. `Moscow, RU` vs `Moscow, ID`). When the geocoder
    returns multiple strong matches the response short-circuits with
    `relay_to_user: false` and the LLM is expected to ask the user
    which place before re-calling.
    """
    body = await _get_current_weather_in_city_impl(city, country_code=country_code)
    if body.get("relay_to_user") is False:
        return body  # ambiguity envelope — already final
    return _respond(body)


async def _get_weather_forecast_impl(
    city: str,
    days: int = 7,
    country_code: str | None = None,
) -> dict:
    """Raw daily-forecast fetch — returns either a bare body dict or,
    on geocoder ambiguity (#17), a pre-built envelope with
    `relay_to_user=False`. Decoupled from @mcp.tool so shortcut tools
    can call it without hitting the #19 loop guard twice.
    """
    days = max(1, min(int(days), 16))
    loc, clarify = await _resolve_place(city, country_code=country_code)
    if clarify:
        return clarify
    data = await _fetch_json(
        FORECAST_URL,
        service="Open-Meteo forecast",
        timeout=8.0,
        params={
            "latitude": loc["latitude"],
            "longitude": loc["longitude"],
            "daily": "weather_code,temperature_2m_max,temperature_2m_min,"
                     "precipitation_sum,precipitation_probability_max,wind_speed_10m_max",
            "forecast_days": days,
            "timezone": "auto",
        },
    )
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
        "timezone_id": loc.get("timezone"),
        "days": out,
    }


@mcp.tool()
@_loop_guarded
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

    `city` must be a SINGLE token — place name ("Paris") or postal
    code ("10001"), NEVER a comma-separated address. Put country into
    `country_code` (ISO-3166 alpha-2).
    """
    body = await _get_weather_forecast_impl(city, days=days, country_code=country_code)
    if body.get("relay_to_user") is False:
        return body  # ambiguity envelope
    return _respond(body)


@mcp.tool()
@_loop_guarded
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

    `city` must be a SINGLE token — place name ("Rome") or postal code
    ("00100"), NEVER a comma-separated address. Put country into
    `country_code` (ISO-3166 alpha-2).
    """
    hours = max(1, min(int(hours), 168))
    # `forecast_days` controls how many days Open-Meteo computes;
    # round up from the requested hours so we always have enough rows.
    days = max(1, min((hours + 23) // 24, 7))
    loc, clarify = await _resolve_place(city, country_code=country_code)
    if clarify:
        return clarify
    data = await _fetch_json(
        FORECAST_URL,
        service="Open-Meteo forecast",
        timeout=8.0,
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
    return _respond({
        "location": f"{loc['name']}, {loc['country']}",
        "timezone_id": loc.get("timezone"),
        "hours": out,
    })


@mcp.tool()
@_loop_guarded
async def get_sunrise_sunset(
    city: str,
    date_iso: str | None = None,
    days: int = 1,
    country_code: str | None = None,
) -> dict:
    """Sunrise / sunset / daylight duration for a city for one or more days.

    Answers "when does the sun set in Reykjavik today?", "what's the
    daylight duration on June 21 in Kyiv?". Defaults to today (1 day).

    - `city`: a SINGLE token — place name ("Reykjavik") or postal code,
      NEVER a comma-separated address. Put country into `country_code`.
    - `date_iso`: optional anchor (YYYY-MM-DD). When set, the window
      starts on that date. When `None`, starts today in the city's
      local timezone.
    - `days`: number of consecutive days to return (1-16).
    """
    days = max(1, min(int(days), 16))
    loc, clarify = await _resolve_place(city, country_code=country_code)
    if clarify:
        return clarify
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
    data = await _fetch_json(
        FORECAST_URL,
        service="Open-Meteo forecast",
        timeout=8.0,
        params=params,
    )
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
    return _respond({
        "location": f"{loc['name']}, {loc['country']}",
        "timezone_id": loc.get("timezone"),
        "days": out,
    })


@mcp.tool()
@_loop_guarded
async def get_air_quality(city: str, country_code: str | None = None) -> dict:
    """Current air quality (PM2.5, PM10, ozone, NO2, SO2, CO + AQI) in a city.

    Uses Open-Meteo's air-quality endpoint (no key required). Returns
    both the European AQI and the US AQI scales — the model can pick
    the right one for the user's region. Answers "is the air safe in
    Delhi today?", "should I wear a mask outside?".

    `city` must be a SINGLE token — place name ("Delhi") or postal
    code, NEVER a comma-separated address. Put country into
    `country_code` (ISO-3166 alpha-2).
    """
    loc, clarify = await _resolve_place(city, country_code=country_code)
    if clarify:
        return clarify
    data = await _fetch_json(
        AIR_QUALITY_URL,
        service="Open-Meteo air quality",
        timeout=8.0,
        params={
            "latitude": loc["latitude"],
            "longitude": loc["longitude"],
            "current": "european_aqi,us_aqi,pm10,pm2_5,carbon_monoxide,"
                       "nitrogen_dioxide,sulphur_dioxide,ozone",
            "timezone": "auto",
        },
    )
    cur = data.get("current", {})
    return _respond({
        "location": f"{loc['name']}, {loc['country']}",
        "timezone_id": loc.get("timezone"),
        "time": cur.get("time"),
        "european_aqi": cur.get("european_aqi"),
        "us_aqi": cur.get("us_aqi"),
        "pm2_5_ugm3": cur.get("pm2_5"),
        "pm10_ugm3": cur.get("pm10"),
        "ozone_ugm3": cur.get("ozone"),
        "nitrogen_dioxide_ugm3": cur.get("nitrogen_dioxide"),
        "sulphur_dioxide_ugm3": cur.get("sulphur_dioxide"),
        "carbon_monoxide_ugm3": cur.get("carbon_monoxide"),
    })


@mcp.tool()
@_loop_guarded
async def get_weather_by_coordinates(latitude: float, longitude: float) -> dict:
    """Current weather at raw lat/lon — skips the geocoder entirely.

    **DO NOT INVENT COORDINATES FROM MEMORY.** Never guess lat/lon for
    a named place. Coordinates must come from one of:
      - `detect_my_location_by_ip()` — the caller's own location,
      - `resolve_address(address=...)` — a parsed postal address,
      - `find_place_coordinates(city=...)` — a resolved city/postcode,
      - an explicit value the user pasted from a map app.

    For a named city (e.g. "погода в Bountiful, Utah"), call
    `get_current_weather_in_city(city="Bountiful", country_code="US")`
    directly — it does the geocode for you. This tool exists ONLY for
    unnamed coordinates (lake, trailhead, offshore, or a pre-resolved
    lat/lon).
    """
    if not (-90.0 <= latitude <= 90.0):
        raise ValueError(f"latitude must be in [-90, 90], got {latitude}")
    if not (-180.0 <= longitude <= 180.0):
        raise ValueError(f"longitude must be in [-180, 180], got {longitude}")
    data = await _fetch_json(
        FORECAST_URL,
        service="Open-Meteo forecast",
        timeout=8.0,
        params={
            "latitude": latitude,
            "longitude": longitude,
            "current": "temperature_2m,relative_humidity_2m,apparent_temperature,"
                       "precipitation,weather_code,wind_speed_10m,wind_direction_10m",
            "timezone": "auto",
        },
    )
    cur = data.get("current", {})
    return _respond({
        "latitude": latitude,
        "longitude": longitude,
        "timezone_id": data.get("timezone"),
        "time": cur.get("time"),
        "temperature_c": cur.get("temperature_2m"),
        "apparent_temperature_c": cur.get("apparent_temperature"),
        "humidity_pct": cur.get("relative_humidity_2m"),
        "precipitation_mm": cur.get("precipitation"),
        "wind_kmh": cur.get("wind_speed_10m"),
        "wind_direction_deg": cur.get("wind_direction_10m"),
        "conditions": WEATHER_CODES.get(cur.get("weather_code"), "unknown"),
    })


@mcp.tool()
@_loop_guarded
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

    `city` must be a SINGLE token — place name ("Kyiv") or postal
    code, NEVER a comma-separated address. Put country into
    `country_code` (ISO-3166 alpha-2).
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
    loc, clarify = await _resolve_place(city, country_code=country_code)
    if clarify:
        return clarify
    data = await _fetch_json(
        ARCHIVE_URL,
        service="Open-Meteo archive",
        timeout=10.0,  # archive endpoint is measurably slower than /forecast
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
    return _respond({
        "location": f"{loc['name']}, {loc['country']}",
        "timezone_id": loc.get("timezone"),
        "days": out,
    })


# ── Non-weather knowledge tools ────────────────────────────────────────────
#
# Co-located in the same server because the operator prefers one sidecar
# over two; each docstring is deliberately terse to keep the combined
# tool catalog compact in the model's system prompt.


@mcp.tool()
@_loop_guarded
async def get_wikipedia_summary(title: str, lang: str = "en") -> dict:
    """Short Wikipedia summary (~300 chars) + page URL for a topic.

    Call this whenever the user wants encyclopedic information about
    a place, person, concept or event. English trigger phrases
    include "tell me about X", "who / what is X", "give me background
    on X", "what does Wikipedia say about X". Russian trigger phrases
    include "что википедия говорит про X", "расскажи про X",
    "кто такой X", "что такое X", "расскажи о X".

    `lang` is a Wikipedia language code ("en", "uk", "de", "ru"…).
    Pass your best guess; the server also auto-tries language
    fallbacks based on the title's script — a Cyrillic title with
    `lang="en"` would 404 on en.wiki, so we silently retry on
    ru.wiki / uk.wiki before giving up. The response reports which
    `lang` actually returned the hit. Title can be a name or a
    URL-slug ("Kyiv", "Beverly_Hills", "соль"); pass the literal
    phrase the user said — Wikipedia redirects handle the rest.
    """
    safe_title = title.replace(" ", "_")
    # Build the language chain:
    #   1. Caller's explicit `lang` first (they may know better).
    #   2. Then the script-detected chain for the title — covers the
    #      "Cyrillic title with default lang=en" 404 we hit live on
    #      2026-04-21 ("соль"). `_detect_query_languages` returns an
    #      ordered chain per script (Cyrillic → ru, uk).
    #   3. Always fall through to "en" as last resort — Wikipedia's
    #      English corpus is the largest and has redirect coverage.
    # Deduplicate while preserving first-appearance order.
    seen: set[str] = set()
    chain: list[str] = []
    for candidate in [lang, *_detect_query_languages(title), "en"]:
        if candidate and candidate not in seen:
            seen.add(candidate)
            chain.append(candidate)

    # Wikipedia's REST API enforces its User-Agent ToS — a request
    # without a descriptive UA returns HTTP 403 ("please provide a
    # unique User-Agent with contact info"). Share the same repo-
    # identifying string with radio-browser / OSM for simplicity;
    # Wikipedia only cares that it's descriptive, not exclusive.
    headers = {"Accept": "application/json", "User-Agent": RADIO_BROWSER_UA}

    last_exc: Exception | None = None
    for candidate_lang in chain:
        url = WIKIPEDIA_SUMMARY_URL.format(lang=candidate_lang, title=safe_title)
        try:
            data = await _fetch_json(
                url,
                service="Wikipedia",
                timeout=5.0,
                headers=headers,
            )
        except RuntimeError as e:
            # `_fetch_json` wraps upstream 4xx / 5xx / timeout as
            # RuntimeError. 404 on `/page/summary/<title>` means this
            # lang-wiki doesn't have the article — try the next lang.
            # Keep the last exception so we can raise it if every
            # candidate misses.
            last_exc = e
            continue
        return _respond({
            "title": data.get("title"),
            "description": data.get("description"),
            "extract": data.get("extract"),
            "url": (data.get("content_urls") or {}).get("desktop", {}).get("page"),
            "lang": candidate_lang,
            "lang_chain_tried": chain,
        })

    if last_exc is not None:
        raise last_exc
    # Fell through without any request — means chain was empty, which
    # shouldn't happen because "en" is always appended. Belt-and-
    # suspenders error so an unexpected empty chain doesn't silently
    # return None.
    raise RuntimeError(f"Wikipedia: no lang candidates tried for title={title!r}")


@mcp.tool()
@_loop_guarded
async def get_country_info(country: str) -> dict:
    """Country facts: capital, population, currencies, languages, calling code, neighbours.

    Accepts an ISO-3166-1 alpha-2 or alpha-3 code ("UA", "USA") or a
    plain name ("Ukraine", "United States").
    """
    # Alpha codes hit the /alpha endpoint (exact); free-form names hit /name
    # (may return multiple hits — take the first).
    code = country.strip()
    use_alpha = len(code) in (2, 3) and code.isalpha()
    url = (RESTCOUNTRIES_ALPHA_URL if use_alpha else RESTCOUNTRIES_NAME_URL).format(
        code=code.upper(), name=code,
    )
    data = await _fetch_json(url, service="REST Countries", timeout=5.0)
    hit = data[0] if isinstance(data, list) else data
    currencies = hit.get("currencies") or {}
    languages = hit.get("languages") or {}
    idd = hit.get("idd") or {}
    suffixes = idd.get("suffixes") or [""]
    return _respond({
        "name": (hit.get("name") or {}).get("common"),
        "official_name": (hit.get("name") or {}).get("official"),
        "country_code": hit.get("cca2"),
        "capital": (hit.get("capital") or [None])[0],
        "region": hit.get("region"),
        "subregion": hit.get("subregion"),
        "population": hit.get("population"),
        "area_km2": hit.get("area"),
        "currencies": [
            {"code": k, "name": v.get("name"), "symbol": v.get("symbol")}
            for k, v in currencies.items()
        ],
        "languages": list(languages.values()),
        "calling_code": (idd.get("root") or "") + suffixes[0],
        "borders": hit.get("borders") or [],
        "timezones": hit.get("timezones") or [],
        "flag_emoji": hit.get("flag"),
    })


@mcp.tool()
@_loop_guarded
async def get_public_holidays(country_code: str, year: int | None = None) -> dict:
    """Public / bank holidays for a country in a given year.

    `country_code` is ISO-3166-1 alpha-2 ("UA", "US", "JP"). `year`
    defaults to the current calendar year in UTC.
    """
    if year is None:
        year = datetime.now(_tz.utc).year
    url = HOLIDAYS_URL.format(year=year, country_code=country_code.upper())
    data = await _fetch_json(url, service="Nager public-holidays", timeout=5.0)
    holidays = [
        {
            "date": h.get("date"),
            "local_name": h.get("localName"),
            "name": h.get("name"),
            "is_fixed": h.get("fixed"),
            "is_global": h.get("global"),
            "types": h.get("types") or [],
        }
        for h in data or []
    ]
    return _respond({
        "country_code": country_code.upper(),
        "year": year,
        "holidays": holidays,
    })


@mcp.tool()
@_loop_guarded
async def convert_currency(
    amount: float,
    from_currency: str,
    to_currency: str,
) -> dict:
    """Convert an amount between two fiat currencies at today's rate.

    `from_currency` / `to_currency` are ISO-4217 codes ("USD", "EUR",
    "UAH"). Uses open.er-api.com (no key, daily rates). Answers
    "how much is 50 USD in EUR?".
    """
    base = from_currency.strip().upper()
    target = to_currency.strip().upper()
    data = await _fetch_json(
        CURRENCY_URL.format(base=base), service="ExchangeRate-API", timeout=5.0
    )
    if data.get("result") != "success":
        raise ValueError(f"Currency API error: {data.get('error-type') or 'unknown'}")
    rates = data.get("rates") or {}
    if target not in rates:
        raise ValueError(f"Unknown target currency: {target}")
    rate = rates[target]
    return _respond({
        "amount": amount,
        "from": base,
        "to": target,
        "rate": rate,
        "converted": round(amount * rate, 4),
        "rate_date": data.get("time_last_update_utc"),
    })


# ── calculator (safe AST walker) ─────────────────────────────────────────
#
# Small-LLM arithmetic is reliably wrong past 2-3 digit × 2-3 digit and
# anything chained ("A plus B percent of C"). A trivial tool closes the
# accuracy gap. AST-walker impl (NOT `eval()`) so the tool surface
# doesn't accidentally become a Python-code sandbox escape — we allow
# numeric literals, the six arithmetic operators, parentheses, unary
# ± and a safelisted set of `math.*` functions / constants.
#
# Deliberately narrow: no imports, no attribute access, no assignments,
# no comprehensions, no f-strings, no function definitions. Anything
# not in the whitelist raises a clear ValueError with the expression
# echoed back so the model can self-correct on the next turn.

import ast as _ast
import math as _math

_SAFE_MATH_FUNCS: frozenset[str] = frozenset({
    "sqrt", "cbrt", "log", "log2", "log10", "exp",
    "sin", "cos", "tan", "asin", "acos", "atan", "atan2",
    "sinh", "cosh", "tanh",
    "floor", "ceil", "trunc", "fabs",
    "degrees", "radians",
    "gcd", "lcm", "factorial", "hypot",
})
_SAFE_BUILTINS: dict[str, object] = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "pow": pow,
    "sum": sum,
}
_SAFE_CONSTS: frozenset[str] = frozenset({"pi", "e", "tau", "inf", "nan"})

_SAFE_BINOPS: dict[type, object] = {
    _ast.Add:      lambda a, b: a + b,
    _ast.Sub:      lambda a, b: a - b,
    _ast.Mult:     lambda a, b: a * b,
    _ast.Div:      lambda a, b: a / b,
    _ast.FloorDiv: lambda a, b: a // b,
    _ast.Mod:      lambda a, b: a % b,
    _ast.Pow:      lambda a, b: a ** b,
}
_SAFE_UNARYOPS: dict[type, object] = {
    _ast.USub: lambda v: -v,
    _ast.UAdd: lambda v: +v,
}


def _safe_eval(expression: str) -> float | int:
    """Evaluate `expression` as a numeric formula. Whitelist-only AST walk.

    Raises ValueError on any node shape outside the whitelist
    (identifiers, function calls, operators). The error message names
    the specific forbidden construct so the model can fix the call
    rather than retrying an identical broken expression (which the
    #19 loop detector would otherwise short-circuit).
    """
    # Calculator surfaces everywhere use `^` for power; Python parses it
    # as bitwise XOR which raises on floats and has *lower* precedence
    # than `*`. Rewriting before parsing keeps `pi * r^2` meaning `pi *
    # r**2`, which is what LLMs generate when asked for circle area.
    # The AST walk still whitelists Pow, so security is unchanged.
    expression = expression.replace("^", "**")
    try:
        tree = _ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"syntax error in expression: {exc.msg}") from None
    return _safe_eval_node(tree.body)


def _safe_eval_node(node: _ast.AST) -> float | int:
    if isinstance(node, _ast.Constant):
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        raise ValueError(f"only numeric literals allowed, got {type(node.value).__name__}")
    if isinstance(node, _ast.BinOp):
        op = _SAFE_BINOPS.get(type(node.op))
        if op is None:
            raise ValueError(f"operator not allowed: {type(node.op).__name__}")
        return op(_safe_eval_node(node.left), _safe_eval_node(node.right))
    if isinstance(node, _ast.UnaryOp):
        op = _SAFE_UNARYOPS.get(type(node.op))
        if op is None:
            raise ValueError(f"unary op not allowed: {type(node.op).__name__}")
        return op(_safe_eval_node(node.operand))
    if isinstance(node, _ast.Name):
        if node.id in _SAFE_CONSTS:
            return getattr(_math, node.id)
        raise ValueError(
            f"unknown name {node.id!r}; constants allowed: {sorted(_SAFE_CONSTS)}"
        )
    if isinstance(node, _ast.Call):
        func = node.func
        if not isinstance(func, _ast.Name):
            raise ValueError("only bare function calls allowed (no attribute access)")
        name = func.id
        # Validate the function name BEFORE evaluating args. An unknown
        # `__import__(...)` with a string literal arg would otherwise
        # error on the string-not-numeric check first, leaking an
        # error message that blames the arg when the real issue is the
        # function. Order matters for clarity, not security — either
        # rejection stops the walk.
        if name not in _SAFE_MATH_FUNCS and name not in _SAFE_BUILTINS:
            raise ValueError(
                f"unknown function {name!r}; math funcs: {sorted(_SAFE_MATH_FUNCS)}; "
                f"builtins: {sorted(_SAFE_BUILTINS)}"
            )
        if node.keywords:
            raise ValueError(f"keyword arguments not allowed for {name!r}")
        args = [_safe_eval_node(a) for a in node.args]
        if name in _SAFE_MATH_FUNCS:
            return getattr(_math, name)(*args)
        return _SAFE_BUILTINS[name](*args)
    raise ValueError(f"expression node not allowed: {type(node).__name__}")


@mcp.tool()
@_loop_guarded
async def calculate(expression: str) -> dict:
    """Evaluate an arithmetic expression. Use for any math the user asks.

    **Use this for ANY numeric answer you would otherwise compute from
    memory.** Small-LLM arithmetic is reliably wrong on 4+ digit
    multiplications, chained percentages, and non-trivial rounding.
    Cheaper to call this than to guess.

    Supported:
      - Arithmetic: `+ - * / // % **`, unary `-`, parentheses.
        `^` also works as a power shortcut (auto-rewritten to `**`
        with correct precedence — `pi * r^2` → `pi * r**2`).
      - Constants: `pi`, `e`, `tau`, `inf`, `nan`.
      - `math.*` functions: `sqrt`, `cbrt`, `log` (natural),
        `log2`, `log10`, `exp`, `sin`, `cos`, `tan`, `asin`, `acos`,
        `atan`, `atan2`, `sinh`, `cosh`, `tanh`, `floor`, `ceil`,
        `trunc`, `fabs`, `degrees`, `radians`, `gcd`, `lcm`,
        `factorial`, `hypot`.
      - Builtins: `abs`, `round`, `min`, `max`, `pow`, `sum`.

    Example expressions by shape:
      - Plain:          `3847 * 29`, `(12345 + 678) / 9`
      - Percentages:    `2450 * 0.15`, `2450 * 15 / 100`
      - Chained:        `(300 + 50) * 1.08 / 2`
      - Powers/roots:   `2 ** 10`, `sqrt(2450)`, `cbrt(27)`
      - Logs:           `log10(1000)`, `log(8, 2)`
      - Trig:           `sin(radians(30))`, `atan2(1, 1)`

    Geometry patterns (no dedicated functions — just express):
      - Circle area:        `pi * r**2`      e.g. `pi * 5**2`
      - Circle circum.:     `2 * pi * r`     e.g. `2 * pi * 5`
      - Triangle area:      `0.5 * b * h`
      - Rectangle area:     `w * h`
      - Sphere volume:      `(4/3) * pi * r**3`
      - Cylinder volume:    `pi * r**2 * h`
      - Cube volume:        `s**3`
      - Hypotenuse:         `hypot(a, b)`
      - Distance (2D):      `hypot(x2-x1, y2-y1)`
      - Angle deg→rad:      `radians(30)`
      - Angle rad→deg:      `degrees(pi / 4)`

    NOT supported — errors with a clear message: symbolic math (no
    `x` / `y` / unresolved names), units (`5 meters` — strip the unit
    yourself), currency (use the currency-conversion tool),
    attribute access (`math.sqrt(2)` — write `sqrt(2)` instead),
    keyword arguments, lists / sequences.

    Args:
      expression: formula, no leading `=`, no units.
    """
    try:
        value = _safe_eval(expression)
    except ValueError as exc:
        return _respond(
            {"expression": expression, "error": str(exc)},
            relay_to_user=False,
            guidance="Expression rejected. Rewrite using plain arithmetic and listed functions/constants, or ask the user to clarify. Do not retry an identical expression.",
        )
    except ZeroDivisionError:
        return _respond(
            {"expression": expression, "error": "division by zero"},
            relay_to_user=False,
            guidance="Division by zero is undefined. Ask the user whether they meant something else.",
        )
    return _respond({"expression": expression, "result": value})


@mcp.tool()
@_loop_guarded
async def list_radio_stations(
    country: str | None = None,
    tag: str | None = None,
    language: str | None = None,
    limit: int = 5,
) -> dict:
    """Find internet-radio stations by country / tag / language.

    Answers "radio stations in Ukraine", "Ukrainian-language radio",
    "jazz stations". Pass at least one filter; `limit` caps the
    result count (1-20). Data from radio-browser.info (volunteer
    community catalogue). **Streams only** — FM/AM frequencies are
    regulator-specific and out of scope; this tool returns internet
    radio URLs.

    **`country` accepts two shapes** — prefer ISO for correctness:
      - ISO-3166-1 alpha-2 code ("US", "UA", "DE") — routed to
        radio-browser's exact-code endpoint, returns only stations
        actually registered in that country.
      - Full English country name ("United States", "Ukraine",
        "Germany") — routed to radio-browser's name endpoint, which
        does a FUZZY substring match. `"US"` passed as a name would
        wrongly match "Russian Federation", "Australia" etc. —
        that's why the 2-letter path exists and is preferred.

    **`language` is a full English name**, not an ISO code:
      - Correct: `"russian"`, `"ukrainian"`, `"english"`, `"spanish"`.
      - Wrong: `"ru"`, `"uk"`, `"en"` — radio-browser indexes by the
        spelled-out name and an ISO code returns empty results.
    """
    limit = max(1, min(int(limit), 20))
    if not any([country, tag, language]):
        raise ValueError(
            "At least one filter (country, tag, or language) is required to "
            "avoid dumping the full catalogue."
        )
    # radio-browser has dedicated /bycountry, /bytag, /bylanguage endpoints;
    # combine filters by intersecting client-side — the API accepts only one
    # selector per call. Start with the most specific filter.
    #
    # For `country` we split the path on input shape: a bare 2-letter
    # alpha token ("US", "ua", "De") hits the exact-code endpoint so the
    # result set is actually that country. Anything else goes to the
    # name endpoint. Without this split, "US" passes as a name and the
    # fuzzy substring match drags back "Russian Federation",
    # "Australia", etc. (observed live during a 2026-04-20 mcphost
    # session, issue #13).
    if country:
        token = country.strip()
        if len(token) == 2 and token.isalpha():
            quoted = httpx.QueryParams({'q': token.upper()})['q']
            path = f"/stations/bycountrycodeexact/{quoted}"
        else:
            quoted = httpx.QueryParams({'q': token})['q']
            path = f"/stations/bycountry/{quoted}"
    elif tag:
        path = f"/stations/bytag/{httpx.QueryParams({'q': tag})['q']}"
    else:
        path = f"/stations/bylanguage/{httpx.QueryParams({'q': language})['q']}"
    headers = {"User-Agent": RADIO_BROWSER_UA, "Accept": "application/json"}
    # Iterate through every mirror — if one is down, fall through. Each
    # mirror serves the same data pool so the result shape is identical.
    data = await _fetch_json(
        [f"{base}{path}" for base in RADIO_BROWSER_MIRRORS],
        service="radio-browser",
        timeout=6.0,
        headers=headers,
    )
    filtered = data
    if tag:
        tag_low = tag.lower()
        filtered = [s for s in filtered if tag_low in (s.get("tags") or "").lower()]
    if language:
        lang_low = language.lower()
        filtered = [s for s in filtered if lang_low in (s.get("language") or "").lower()]
    # Sort by click count desc so the most-listened-to stations are first.
    filtered.sort(key=lambda s: s.get("clickcount") or 0, reverse=True)
    # Minimal per-station payload — the chat model only needs
    # enough to pick one and hand the user a playable URL. Homepage,
    # tags, codec, bitrate, language were live bloat: 8 fields × 10
    # stations used to clock 3-5K tokens per response, enough to
    # budget out the context on follow-up turns. Names are also
    # capped because a few stations ship 100+ char joke titles.
    stations = [
        {
            "name": (s.get("name") or "")[:60],
            "country": s.get("country"),
            "url": s.get("url_resolved") or s.get("url"),
        }
        for s in filtered[:limit]
    ]
    return _respond({
        "filters": {"country": country, "tag": tag, "language": language},
        "count": len(stations),
        "stations": stations,
    })


LIVENESS_PATHS = frozenset({"/healthz", "/livez"})
READINESS_PATH = "/readyz"
HEALTH_PATHS = LIVENESS_PATHS | {READINESS_PATH}


async def _readiness_check() -> tuple[bool, dict]:
    """Probe Open-Meteo's geocoder to decide whether we can serve work.

    The sidecar is a thin wrapper over Open-Meteo — if the geocoder
    is unreachable, every real tool call will fail the same way, so
    flipping `/readyz` to 503 lets k8s / the orchestrator surface
    that signal instead of us faking readiness. Kept ~2 s total so
    a probe doesn't wedge if upstream is genuinely hanging.

    Returns `(ok, checks_payload)` where `checks_payload` is the
    per-dependency dict that goes into the JSON response.
    """
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            r = await client.get(
                GEOCODE_URL,
                params={"name": "Kyiv", "count": 1},
            )
            r.raise_for_status()
        return True, {"open_meteo": "ok"}
    except Exception as exc:
        return False, {"open_meteo": f"{type(exc).__name__}"}


def _build_http_app():
    """Build the Starlette app with probe endpoints + optional bearer auth.

    Extracted from `_run_http_with_auth` so unit tests can drive the
    full middleware stack through an ASGI transport without binding a
    port. Middleware layering is load-bearing: the probe middleware
    MUST run BEFORE bearer auth so probes never need a token
    (Starlette stacks middleware LIFO on the incoming side — last
    added = outermost).
    """
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

    class HealthzMiddleware(BaseHTTPMiddleware):
        """Kubernetes-style probe paths, split by k8s convention.

        - `/healthz`, `/livez` — liveness. Process is up, event loop
          responsive. No I/O — returns 200 immediately so a hung
          upstream can never fail-loop the pod restart.
        - `/readyz` — readiness. Active probe against Open-Meteo's
          geocoder; 503 with details if upstream is unreachable.
          k8s drains traffic until it clears. Bypasses bearer-auth
          so k8s probes never need a token.

        All three routes short-circuit before the MCP initialize
        handshake, which is too expensive to do per probe tick.
        """
        async def dispatch(self, request, call_next):
            path = request.url.path
            if path in LIVENESS_PATHS:
                return JSONResponse({
                    "status": "ok",
                    "service": "mcp-weather",
                    "probe": "liveness",
                })
            if path == READINESS_PATH:
                ok, checks = await _readiness_check()
                return JSONResponse(
                    {
                        "status": "ok" if ok else "not_ready",
                        "service": "mcp-weather",
                        "probe": "readiness",
                        "checks": checks,
                    },
                    status_code=200 if ok else 503,
                )
            return await call_next(request)

    # Added LAST so it wraps bearer auth — incoming request hits
    # HealthzMiddleware first, short-circuits probe paths before auth
    # ever runs.
    app.add_middleware(HealthzMiddleware)
    return app


def _run_http_with_auth() -> None:
    import uvicorn

    uvicorn.run(_build_http_app(), host=HOST, port=PORT, log_level="info")


# ── outputSchema A/B experiment (no community consensus) ──────────────────
#
# MCP spec 2025-06-18 added `outputSchema` + `structuredContent` for tools.
# Community is split — modelcontextprotocol discussion #1121 has devs
# reporting "notable improvement" on plain text and others defending
# schemas. Rather than pick on vibes we run it through the eval matrix.
#
# MCP_OUTPUT_SCHEMA="on"  → every tool declares the shared envelope schema
#                           below (relay_to_user + guidance required, extra
#                           body fields permitted).
# MCP_OUTPUT_SCHEMA="off" → no outputSchema on any tool (current default).
#
# Set at module import time; the env var is read once and baked into the
# tool catalog. The eval workflow (.github/workflows/eval.yml) can flip
# it per-matrix-row so 7b and 14b each get measured on both settings.

_OUTPUT_SCHEMA_MODE = os.getenv("MCP_OUTPUT_SCHEMA", "off").strip().lower()

# Shared schema: every tool's response conforms to the envelope from #18.
# `additionalProperties: true` lets each tool's body shape vary (weather
# fields, radio stations, wikipedia snippets, ...) without having to
# declare per-tool schemas.
_ENVELOPE_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "relay_to_user": {
            "type": "boolean",
            "description": (
                "True = model can answer the user from this body. "
                "False = model MUST clarify with the user before replying "
                "(ambiguous input, duplicate call, missing context)."
            ),
        },
        "guidance": {
            "type": "string",
            "description": (
                "Plain-English one-liner instructing the LLM what to do "
                "with the body. Short on purpose — follow it literally."
            ),
        },
    },
    "required": ["relay_to_user", "guidance"],
    "additionalProperties": True,
}


def _apply_output_schema_experiment() -> None:
    """When MCP_OUTPUT_SCHEMA=on, attach _ENVELOPE_OUTPUT_SCHEMA to every
    registered tool so the MCP catalog advertises the envelope contract
    via spec 2025-06-18's `outputSchema` field.

    Runs after all @mcp.tool decorators have registered, so every Tool
    object is in `mcp._tool_manager._tools` by the time we iterate.
    """
    if _OUTPUT_SCHEMA_MODE != "on":
        return
    for tool in mcp._tool_manager._tools.values():
        tool.output_schema = _ENVELOPE_OUTPUT_SCHEMA


_apply_output_schema_experiment()


# ── Docstring verbosity A/B (terse vs verbose) ─────────────────────────────
#
# Source docstrings live in `terse` mode: short, bullet-style, no long
# examples — optimised for small-model prefill speed on CPU (the top-3
# offenders used to cost ~550 extra tokens per request). Bigger models
# may benefit from the verbose versions kept below; flip the env var to
# opt in. Same A/B pattern as MCP_OUTPUT_SCHEMA so the eval matrix can
# measure which is better for which model tier.
#
# MCP_DOCSTRING_MODE="terse"   → source defaults (current behaviour).
# MCP_DOCSTRING_MODE="verbose" → restore pre-trim descriptions for the
#                                three tools where we cut the most.

_DOCSTRING_MODE = os.getenv("MCP_DOCSTRING_MODE", "terse").strip().lower()

# Pre-trim descriptions, verbatim from before the 2026-04-21 trim.
# Keyed by tool name; absent keys fall through to the source docstring.
_DOCSTRINGS_VERBOSE: dict[str, str] = {
    "find_place_coordinates": (
        "Resolve a city name or postal code to lat/lon, country and timezone.\n\n"
        "**Argument contract — read before calling:** `city` MUST be a "
        "single canonical token. Valid shapes:\n"
        "  - a plain place name: `\"Kyiv\"`, `\"Paris\"`, `\"Bountiful\"`.\n"
        "  - a numeric postal code: `\"90210\"`, `\"10001\"`, `\"84010\"`.\n"
        "Invalid — Open-Meteo matches the whole string literally and will "
        "return no result:\n"
        "  - `\"Paris, France\"` — country goes into `country_code`, not the name.\n"
        "  - `\"Bountiful, Utah, 84010\"` — address string, never works.\n"
        "  - `\"Kyiv, Ukraine\"` — comma kills the match.\n\n"
        "Postal-code support varies: US/DE/FR zipcodes resolve cleanly; "
        "UK postcodes (`\"SW1A 1AA\"`) are **not** indexed and return empty. "
        "For a postal code always pass `country_code` too — `\"10001\"` "
        "matches both New York, US and Troyes, FR.\n\n"
        "`country_code` is an optional ISO-3166-1 alpha-2 hint (\"US\", \"UA\", "
        "\"GB\") to disambiguate homonyms like \"Moscow, RU\" vs \"Moscow, ID\". "
        "The response includes `admin1` (state/region) and `postcodes` so "
        "the caller can verify the right place was picked.\n\n"
        "When multiple comparable places match (e.g. \"Springfield\" without a "
        "country, \"Moscow\" without country_code, the 5 Bountifuls in different "
        "US states), the response flips `relay_to_user` to `false` and lists "
        "candidates in `guidance` — the LLM must ask the user which one to "
        "pick instead of silently committing to top-1."
    ),
    "search_places": (
        "Return every geocoding candidate for an ambiguous query.\n\n"
        "Deliberately general-purpose: the Open-Meteo geocoder returns not "
        "just towns but also mountains, lakes, parks, islands, neighborhoods "
        "and airports bearing the same name. Each candidate carries a "
        "`feature_type` human label (\"city\", \"mountain\", \"lake\", "
        "\"neighborhood\", \"park\", \"peak\", \"island\", \"airport\", …) so "
        "the caller can match the user's intent (\"weather on Mt. Bountiful\" "
        "vs \"weather in Bountiful\" are legitimately different places).\n\n"
        "Parameters:\n"
        "- `query`: a SINGLE canonical token — place name (`\"Springfield\"`, "
        "`\"Bountiful\"`) OR postal code (`\"84010\"`). Do NOT pass a "
        "comma-separated address (`\"Bountiful, Utah, 84010\"`) — Open-Meteo "
        "matches the whole string literally and will return empty. "
        "Put country into `country_code` below.\n"
        "- `country_code`: ISO-3166-1 alpha-2 hint (\"US\", \"UA\"). Narrows the "
        "candidate pool server-side.\n"
        "- `feature_types`: optional allowlist of human labels "
        "(e.g. `[\"city\", \"village\"]` for only populated places, or "
        "`[\"mountain\", \"peak\", \"hill\"]` for only high ground). Pass "
        "`None` / empty to keep every feature type in the results.\n"
        "- `limit`: caps the list length (1-10, default 5).\n\n"
        "Use this tool whenever the user's request is vague — either because "
        "the name is ambiguous (\"Springfield\"), the intent isn't clear "
        "(town vs mountain vs lake), or the first hit's `feature_type` "
        "doesn't match what the user seems to want. The caller can then "
        "surface the disambiguation choice to the user or pick one on their "
        "behalf.\n\n"
        "When the query has a comma AND no candidates are found the response "
        "grows a `hint` field telling the caller how to rewrite the query."
    ),
    "resolve_address": (
        "Normalise a free-form postal address into structured components.\n\n"
        "**When to use**: the user's text really is an address — multiple "
        "comma-separated parts, a street number, a postcode, a full "
        "\"street, city, region, country\" tail. Classic triggers:\n"
        "  - `\"Bountiful, Utah, 84010\"` — city + state + zip.\n"
        "  - `\"221B Baker Street, London\"` — street-level.\n"
        "  - `\"Бульвар Лобановського, 23, Київ\"` — Cyrillic full address.\n"
        "  - `\"横浜市中区山下町\"` — non-Latin address.\n\n"
        "**When NOT to use**: a single canonical token (plain city name "
        "OR bare postcode) — those belong in `find_place_coordinates` / "
        "`search_places` / `get_current_weather_in_city` directly. This "
        "tool is the escape hatch for the OTHER case.\n\n"
        "**Response** carries latitude/longitude AND English-normalised "
        "`city` + `country_code` you can feed straight into the weather / "
        "time tools. For example:\n\n"
        "    1. `resolve_address(\"Bountiful, Utah, 84010\")`\n"
        "       → `{\"city\": \"Bountiful\", \"country_code\": \"US\",\n"
        "            \"latitude\": 40.89, \"longitude\": -111.88, …}`\n"
        "    2. `get_current_weather_in_city(city=\"Bountiful\",\n"
        "                                    country_code=\"US\")`.\n\n"
        "Alternatively for current weather only, one step:\n"
        "    `get_weather_by_coordinates(latitude, longitude)`.\n\n"
        "`country_code` (optional, ISO-3166-1 alpha-2) narrows the "
        "geocoder when the address is ambiguous across borders. Leave "
        "unset if unsure.\n\n"
        "Sources: Nominatim (OSM) primary, Photon fallback; both "
        "keyless. See server.py module docstring for the fallback chain."
    ),
}


def _apply_docstring_experiment() -> None:
    """When MCP_DOCSTRING_MODE=verbose, override the terse source
    docstrings with the pre-trim versions for the tools in
    `_DOCSTRINGS_VERBOSE`. Tool.description is what FastMCP forwards
    in `tools/list` responses, so overriding it is enough to change
    what the LLM sees.
    """
    if _DOCSTRING_MODE != "verbose":
        return
    for name, description in _DOCSTRINGS_VERBOSE.items():
        tool = mcp._tool_manager._tools.get(name)
        if tool is not None:
            tool.description = description


_apply_docstring_experiment()


# ── Experimental router mode (MCP_ROUTER_MODE=list_changed) ─────────────────
#
# See `docs/tool-catalog-scaling.md` for the design context. Goal of
# this block is NOT to be the production answer — it's a probe of
# whether `notifications/tools/list_changed` reliably reaches the real
# clients we use (mcphost, Open WebUI). If yes, we can narrow the
# per-turn tool catalog from 23 tools to ~5-10 and free 2-3K tokens of
# prefill. If not, we fall back to the first-message heuristic also
# described in that doc.
#
# Shape:
#   1. Initial `tools/list` → returns only `select_domain(domain)`.
#   2. Model calls `select_domain(domain="weather")`.
#   3. Handler updates process-global `_ROUTER_STATE["active"]` and
#      pushes `send_tool_list_changed()` on the current session.
#   4. Client (if compliant) re-fetches `tools/list` and sees the
#      domain subset + `select_domain` (so cross-domain switches
#      mid-session stay possible).
#
# Single process-global state is fine for the L4 single-user testing
# scenario; for multi-tenant it would need per-session storage.

_ROUTER_DOMAINS: dict[str, tuple[str, ...]] = {
    "weather": (
        "get_weather_outside_right_now",
        "get_weather_forecast_for_today",
        "get_weather_forecast_for_tomorrow",
        "get_current_weather_in_city",
        "get_weather_forecast",
        "get_hourly_forecast",
        "get_weather_by_coordinates",
        "get_historical_weather",
        "get_sunrise_sunset",
        "get_air_quality",
    ),
    "location": (
        "find_place_coordinates",
        "search_places",
        "resolve_address",
        "detect_my_location_by_ip",
        "lookup_ip_geolocation",
    ),
    "time": (
        "get_current_time_where_i_am",
        "get_current_time_in_city",
        "get_current_date",
    ),
    "knowledge": (
        "get_wikipedia_summary",
        "get_country_info",
        "get_public_holidays",
        "convert_currency",
        "list_radio_stations",
        "calculate",
    ),
}

_ROUTER_STATE: dict[str, str | None] = {"active": None}


def _install_router() -> None:
    """Wire the chosen router variant onto the FastMCP instance.

    Idempotent — dispatches on ``MCP_ROUTER_MODE``:
      - ``off`` (default): no-op, monolith as advertised.
      - ``list_changed``: single ``select_domain`` entry point + filtered
        ``list_tools`` that flips on `send_tool_list_changed`. See
        `docs/tool-catalog-scaling.md` for the theory and known client
        limitations (mcphost / Open WebUI don't honour the notification).
      - ``fat_tools``: four fat domain-tools (`weather` / `geo` /
        `knowledge` / `radio`) dispatching via `action` enum + per-action
        named kwargs. Works with static-catalog clients — no protocol
        magic. Catalog ~1885 tokens (−64% vs monolith) on the current
        23-tool set.
      - ``fat_tools_lean``: same four tools but signature collapses to
        `(action, params={})` — drops the pydantic-generated
        nullability schema on every optional kwarg (fat_tools' 1885
        tokens include ~1000 of that shell). Target catalog
        ~900-1000 tokens. Tradeoff: model sees `params` as a plain
        object with no type hints — has to learn the per-action
        param shape from the docstring alone.
    """
    if ROUTER_MODE == "off":
        return
    if ROUTER_MODE == "fat_tools":
        _install_fat_tools_mode()
        return
    if ROUTER_MODE == "fat_tools_lean":
        _install_fat_tools_lean_mode()
        return
    if ROUTER_MODE != "list_changed":
        raise RuntimeError(
            f"MCP_ROUTER_MODE={ROUTER_MODE!r} unrecognised; expected "
            "one of: off, list_changed, fat_tools, fat_tools_lean."
        )

    # Sanity-check the domain map against the registered tool names so
    # a rename in one place + forgotten update here fails loudly at
    # boot rather than silently hiding a tool from the router.
    all_registered = {t.name for t in mcp._tool_manager.list_tools()}
    mapped = {name for names in _ROUTER_DOMAINS.values() for name in names}
    missing = mapped - all_registered
    unmapped = all_registered - mapped
    if missing:
        raise RuntimeError(f"router domain map references unknown tools: {sorted(missing)}")
    if unmapped:
        raise RuntimeError(
            f"router domain map leaves tools unreachable: {sorted(unmapped)} "
            "— add them to _ROUTER_DOMAINS or this server is only partially routable."
        )

    valid_domains = tuple(_ROUTER_DOMAINS.keys())

    @mcp.tool()
    async def select_domain(
        domain: Literal["weather", "location", "time", "knowledge"],
        ctx: Context,
    ) -> dict:
        """Route this session to a tool subset. **Call once per user intent.**

        Picks which family of tools becomes available to you:
          - `weather` — current / forecast / historical / air / sunrise.
          - `location` — geocoding, address parse, IP geolocation.
          - `time`     — current date/time here or in a named city.
          - `knowledge` — wikipedia, country info, holidays, currency,
                          radio stations.

        After you call this, the tool list updates (the client will
        re-fetch it) and the domain's tools appear. Switch domains
        mid-conversation by calling `select_domain` again. When the
        user's next turn is clearly in a different family, switch
        before calling the actual tool.
        """
        prev = _ROUTER_STATE["active"]
        _ROUTER_STATE["active"] = domain
        try:
            await ctx.session.send_tool_list_changed()
        except Exception as exc:
            # Don't fail the tool call — the state change still counts;
            # we just surface that the notification didn't land. On
            # non-stateful sessions this is expected.
            return {
                "active_domain": domain,
                "previous_domain": prev,
                "list_changed_notification": f"failed: {type(exc).__name__}",
                "guidance": "Domain selected but the client may not auto-refresh. Try calling a domain tool directly; if it's not visible, call select_domain again.",
            }
        return {
            "active_domain": domain,
            "previous_domain": prev,
            "list_changed_notification": "sent",
            "guidance": "Client should now re-fetch tool list. Proceed with the domain's tool.",
        }

    # Wrap the FastMCP list_tools handler. The low-level Server stores
    # whatever we register last; re-registering overwrites cleanly.
    _base_list_tools = mcp.list_tools

    async def _router_list_tools():
        all_tools = await _base_list_tools()
        active = _ROUTER_STATE["active"]
        if active is None:
            # No domain selected yet — expose only the router entry
            # point. This is the main-menu state.
            return [t for t in all_tools if t.name == "select_domain"]
        visible = {"select_domain", *_ROUTER_DOMAINS.get(active, ())}
        return [t for t in all_tools if t.name in visible]

    mcp._mcp_server.list_tools()(_router_list_tools)


# Names of the fat domain-tools registered by `install_fat_tools` in
# fat_tools.py. Kept at module scope so the `_install_fat_tools_mode`
# filter knows which of the registered tools to keep visible.
_FAT_TOOL_NAMES: frozenset[str] = frozenset(
    {"weather", "geo", "knowledge", "radio"}
)


def _install_fat_tools_mode() -> None:
    """Register the 4 fat domain-tools and hide the 23 narrow ones.

    Works with static-catalog clients (mcphost, Open WebUI) — no
    `notifications/tools/list_changed` involved. Clients see a stable,
    short tool list the entire session; the routing logic lives inside
    each fat tool, dispatched by its `action` argument.
    """
    import fat_tools  # local import — only loaded when this mode is on.

    fat_tools.install_fat_tools(mcp)

    # Enforce that each narrow tool in _ROUTER_DOMAINS has a fat-tool
    # action that covers it. The map is already validated in the
    # list_changed path; reusing it here gives us the same "renamed a
    # narrow tool but forgot to update the router" guard for free.
    all_registered = {t.name for t in mcp._tool_manager.list_tools()}
    required = _FAT_TOOL_NAMES | {
        name for names in _ROUTER_DOMAINS.values() for name in names
    }
    missing = required - all_registered
    if missing:
        raise RuntimeError(
            f"fat-tools mode missing expected registrations: {sorted(missing)}"
        )

    _base_list_tools = mcp.list_tools

    async def _fat_list_tools():
        # Hide every narrow tool — the fat 4 are the only surface the
        # model is allowed to see in this mode.
        all_tools = await _base_list_tools()
        return [t for t in all_tools if t.name in _FAT_TOOL_NAMES]

    mcp._mcp_server.list_tools()(_fat_list_tools)


def _install_fat_tools_lean_mode() -> None:
    """Register the 4 lean fat-domain tools (signature: action + params dict).

    Same list_tools-filter machinery as ``_install_fat_tools_mode``, just
    registering the `fat_tools_lean.*` variants (which share the fat-
    tool names `weather` / `geo` / `knowledge` / `radio`). Tool-
    manager registration is mutually exclusive — lean and non-lean
    can't both be active because FastMCP would reject the name clash.
    """
    import fat_tools_lean

    fat_tools_lean.install_fat_tools_lean(mcp)

    all_registered = {t.name for t in mcp._tool_manager.list_tools()}
    required = _FAT_TOOL_NAMES | {
        name for names in _ROUTER_DOMAINS.values() for name in names
    }
    missing = required - all_registered
    if missing:
        raise RuntimeError(
            f"fat-tools-lean mode missing expected registrations: {sorted(missing)}"
        )

    _base_list_tools = mcp.list_tools

    async def _lean_list_tools():
        all_tools = await _base_list_tools()
        return [t for t in all_tools if t.name in _FAT_TOOL_NAMES]

    mcp._mcp_server.list_tools()(_lean_list_tools)


_install_router()


if __name__ == "__main__":
    if TRANSPORT == "streamable-http":
        _run_http_with_auth()
    else:
        mcp.run(transport=TRANSPORT)
