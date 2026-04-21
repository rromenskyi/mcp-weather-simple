# Roadmap

Durable snapshot of the backlog so context survives a session reset. Every
bullet links to a GitHub issue with the full reproducer, design, and
acceptance criteria. Ordered roughly by value-per-hour.

This file is tracked in git on purpose — the memory system + open issues
are the canonical sources, this is the human-readable cross-link for
someone opening the repo cold.

## Tool-quality pipeline — make the MCP usable by small LLMs

### [#19 — break tool-call loops (nesting cap + duplicate detector)](https://github.com/rromenskyi/mcp-weather-simple/issues/19)

Single highest-leverage change. Live mcphost session on qwen2.5:7b got
stuck in an **11-call identical-argument loop** across 37 min — same
`list_radio_stations(country="The Russian Federation", language="russian",
limit=5)` every four minutes, never progressing, never asking the user.

Two-part defence:

- **Prompt-side:** MCP server sends `InitializeResult.instructions` with
  "no more than 3 tool calls per user turn; never the same call twice in
  a row; honour `relay_to_user`; ask the user when stuck".
- **Server-side:** duplicate-call detector tracks
  `(tool_name, sorted-kwargs-hash)` per session; second identical call
  short-circuits and returns a `relay_to_user: false` response telling
  the model "you just ran this, use the existing result or ask".

Must normalise kwargs (sort keys) so `{country: X, language: Y}` vs
`{language: Y, country: X}` count as the same call — the live loop
flipped argument order on turn 9.

### [#18 — tool responses carry `relay_to_user` + `guidance`](https://github.com/rromenskyi/mcp-weather-simple/issues/18)

Every tool adds two fields to its return:

- `relay_to_user: bool` — hard contract. When `false`, the LLM MUST
  clarify with the user before answering.
- `guidance: str` — plain English instruction ("Relay directly",
  "Ask the user to pick one: Springfield, IL; Springfield, MA; …",
  "Relay with a caveat that the location was auto-detected from IP").

Plain prose beats a `confidence: enum` — small models follow
instructions better than they interpret controlled vocabularies.
Designed hand-in-hand with #19: `relay_to_user: false` is the single
short-circuit signal the loop detector uses.

Subsumes the ad-hoc `accuracy_warning` field on GeoIP responses and
the `ambiguous` boolean on `search_places`.

### [#17 — surface geocoder ambiguity](https://github.com/rromenskyi/mcp-weather-simple/issues/17)

`_geocode` silently picks `results[0]` even when two real candidates
exist. Makes the LLM confidently wrong:

- `"Springfield"` → top-1 Springfield, MA; user could have meant
  IL / MO / OR.
- `"10001"` → New York, NY; also matches Troyes, FR and Cáceres, ES.
- `"Bountiful"` with `country_code="US"` → UT version; CO also exists.
- `"5 Bountifuls"` problem — intra-country homonyms matter too.

Triggers: top-N cross-country with no `country_code`; top-1 and top-2
populations within 10×; ≥ 2 candidates where `feature_type ∈
{city, village, town}`. Returns `ambiguous: true` + candidate list +
hint to narrow with `country_code` / `admin1`. Final integration:
the ambiguity response becomes one of the cases where
`relay_to_user: false` fires under #18.

### [#14 — `search_places` / `_geocode` empty on comma-separated queries](https://github.com/rromenskyi/mcp-weather-simple/issues/14)

`"Bountiful, Utah, 84010"` returns empty because Open-Meteo treats the
whole string as a literal name match. Fix is **docstring-first** (not
parser hacks): every `city`-accepting tool documents that the argument
must be a single canonical token — a place name OR a postal code — and
country / state go into `country_code` / `admin1`. Optional safety
net: emit a pointed error message when empty + comma is detected, so
the model self-corrects.

### [#13 — `list_radio_stations(country="US")` returns non-US stations](https://github.com/rromenskyi/mcp-weather-simple/issues/13)

radio-browser's `/stations/bycountry/` endpoint expects the full
country name, not an ISO code. The LLM legitimately passed `"US"` and
got back Russian and Australian stations (fuzzy substring match).

Server side: route `"US"` (2-letter alpha) to
`/stations/bycountrycodeexact/{code}`, route `"United States"` to the
existing `/bycountry/` path. Docstring documents both formats work and
recommends ISO. Same nudge for `language` (radio-browser wants
`russian`, not `ru`).

## New capabilities

### [#16 — new `resolve_address` tool via Nominatim + Photon](https://github.com/rromenskyi/mcp-weather-simple/issues/16)

A proper address-normalisation tool so the LLM doesn't have to parse
`"Bountiful, Utah, 84010"` itself. Takes free-form text, returns
`{city, country, country_code, state, postcode, latitude, longitude,
display_name}`. Nominatim (OSM, no-key, 1 rps) primary, Photon
(Komoot/OSM, no limit at our volume) fallback — fits the existing
`_fetch_json` mirror pattern.

Orthogonal to the docstring fix from #14: docstrings teach the LLM to
pass clean tokens; this tool gives it an escape hatch when the input
is genuinely an address. Use either one depending on the shape of the
input.

## Developer ergonomics

### [#15 — eval harness logs tool-call arguments, not just names](https://github.com/rromenskyi/mcp-weather-simple/issues/15)

Today's eval scores top-1 tool name. Misses the "right tool + wrong
args" failure (e.g. `search_places(query="Bountiful, Utah, 84010")` —
correct tool, broken argument). Extend `run_case` to capture
`arguments` and print them in the per-case line and failures table.
Scoring logic unchanged; arguments become a second-signal column.

~10-15 LOC in `tests/integration/eval_tool_calling.py`.

### [#5 — run pytest in CI](https://github.com/rromenskyi/mcp-weather-simple/issues/5)

The unit suite (`tests/test_server.py`, 35 cases, ~1 s) has no
automated runner today. Add a `pytest` job alongside the existing
image-build workflow. Fast, cheap, catches regressions that the eval
harness can't.

### [#3 — decouple shortcut tools from `@mcp.tool` decorator](https://github.com/rromenskyi/mcp-weather-simple/issues/3)

Shortcut tools (`get_weather_outside_right_now`, etc.) currently call
other `@mcp.tool()`-decorated functions as regular Python. Works today
because FastMCP's decorator is transparent; could break on upgrade.
Extract to plain `_impl` helpers. Pure tech debt — do when touching
shortcut code for another reason.

## Experiments (measure before deciding)

### Methodology: env-var toggle + eval matrix A/B

Our pattern for any design question where the community (or we) have
no strong prior:

1. **Gate the variant behind an env var read at server module-load
   time.** Current examples: `MCP_OUTPUT_SCHEMA=on|off` (outputSchema
   experiment, below), `MCP_AUTH_TOKEN` (bearer-auth enable). The
   default must match the current behaviour so unrelated callers
   aren't affected.
2. **Expose the var as a `workflow_dispatch` input with a
   `both` option** in `.github/workflows/eval.yml`. Matrix fans out
   on the new axis so one PR produces numbers for every combination
   in parallel — wall-clock is bounded by the slowest row, not the
   sum. Job names and step-summary headers MUST include the variant
   so results are labelled.
3. **Measure BOTH hit rate AND latency.** The summariser prints a
   per-family `mean / p50 / p95 latency` column next to the rate
   column — some experiments (schema declarations, tool-catalog
   shape changes) can shift inference speed even when tool-selection
   accuracy looks flat.
4. **Promotion rule: ≥ 5 % hit-rate lift at the 7b tier OR a clear
   latency win** to adopt. Otherwise revert and write the result up
   here so the question is closed definitively rather than
   re-surfacing in six months.

The pattern is generic — re-use it for the naming experiment
(`verb_object` vs `domain_prefix`), per-language dataset signals,
any future structured-content tweak.

### Docstring verbosity A/B — terse vs verbose tool descriptions

Same env-var + matrix pattern as `MCP_OUTPUT_SCHEMA`. Hypothesis is
model-tier-dependent: small CPU models benefit from terser
descriptions (less prefill work, crisper signal), bigger models may
want the richer edge-case guidance for disambiguation. No consensus
from the broader community either.

- `MCP_DOCSTRING_MODE=terse` (default): source-level shortened
  descriptions on `find_place_coordinates`, `search_places`,
  `resolve_address` (top-3 chattiest tools, trimmed 2026-04-21).
- `MCP_DOCSTRING_MODE=verbose`: restores pre-trim descriptions for
  those three tools via post-registration override. ~540 extra
  tokens total (~2172 chars).
- Run `workflow_dispatch` with `docstrings=both` (or
  `scripts/eval.sh docstring-ab`) to get a 4-row matrix (2 models ×
  2 modes) at chunk_count=2. Compare per-family rates + latency.

Same promotion rule as other A/Bs: ≥5% hit-rate lift at the 7b
tier OR clear latency win to flip the default.

### outputSchema A/B — MCP spec 2025-06-18, no community consensus

modelcontextprotocol discussion #1121 has devs reporting "notable
improvement" after removing `outputSchema` AND devs defending it. We
run both through the eval matrix to get our own numbers on qwen2.5's
tool-selection surface.

- `MCP_OUTPUT_SCHEMA=off` (default): no `outputSchema` declared on
  any tool — current behaviour, matches the plain-text position.
- `MCP_OUTPUT_SCHEMA=on`: every tool advertises a shared envelope
  schema (`relay_to_user` + `guidance` required,
  `additionalProperties: true` for per-tool body shape). No body-
  schema-per-tool yet — if hit rate or latency shifts meaningfully
  we can chase per-tool schemas next.
- Run `workflow_dispatch` with `output_schema=both` to get
  4-row matrix (2 models × 2 modes) in one pass. Compare per-family
  rates and latency percentiles side-by-side.

### Eval harness v2 (not tracked as a single issue yet — see todo-tomorrow)

- **Per-model timeout bump**: 14b on 4 vCPU needs ~300 s per-request
  vs 7b's 120 s. Current matrix run shows 14b FAILing the threshold
  only because 11/13 failures are `<error: ReadTimeout>` from the
  too-tight ceiling. Excluding timeouts, 14b's real hit rate is
  ≈ 93 % (comparable to 7b post-warm-up). Fix:
  `timeout = 300 if "14b" in model else 120` in `eval_tool_calling.py`.
- **Warm-up timeout**: the warm-up call itself timed out on the 7b
  nightly run ("warm-up failed — proceeding anyway"), leaving cases
  1–3 to hit the cold-start penalty again. Warm-up needs its own
  longer timeout or a simpler payload that doesn't load the full tool
  catalog.
- **Docstring collision**: Wikipedia's expanded «расскажи про X»
  trigger overlaps with `search_places`'s «Springfield в каких
  штатах» intent; both 7b and 14b picked the wrong one on at least
  one case. Fix with explicit "do NOT use this for [adjacent intent]"
  lines in both docstrings.

### Naming experiment: `verb_object` vs `domain_prefix` (not yet an issue)

Industry convention is `verb_object` (`get_current_weather_in_city`),
matches LLM training distribution. Domain-prefix (`weather_current`)
might be shorter but likely hurts small-model selection. Don't
decide on vibes — run the eval matrix on a `experiment/domain-prefix`
branch, compare per-family rates against baseline. If >5 % lift at
the 7b tier, adopt; otherwise revert and write the result up so the
question doesn't re-open.

## Upstream / infrastructure

### [open-webui/open-webui#23907](https://github.com/open-webui/open-webui/issues/23907) — OWUI 0.8.12 native FC bug

OWUI doesn't forward `tools` to Ollama when `function_calling: native`
is persisted on a custom model variant. `mcphost` bypass works; eval
harness bypasses OWUI entirely. Wait for the OWUI dev branch (already
on 0.9.0) to publish a `:latest` image, then retest.

### GPU arrival

On i7 CPU, the matrix run takes ~40 min (bounded by 14b). On a
mid-range consumer GPU (RTX 3060 12 GB / RTX 4070 Ti Super 16 GB /
used 3090 24 GB) the same run should finish in 3–5 min, unlocking
"run this eval on every PR" cadence instead of just nightly.

## Ideas backlog — RV / travel surface (not yet tracked as issues)

Suggestions from a Codex review pass focused on the user's RV use case
(Utah-based, currently boondocking / route-planning), plus two generic
helpers. Each would be a standalone tool on the existing
`_fetch_json` + mirror pattern — no new infra. Score with the eval
harness before promoting any of these to issues.

- **`get_tonight_rv_risk(location)`** — rolls overnight low temp,
  wind gust max, precip probability, and AQI into a single
  freeze/wind/rain/air risk score. Motivation: the user keeps asking
  "can I sleep here tonight?" and composing that from four separate
  tool calls is exactly the nesting pattern #19 is trying to avoid.
- **`best_departure_window(location, next_hours=24)`** — scans the
  hourly forecast for the calmest 2–4 h driving window (low wind,
  dry, daylight). Same composition argument as above.
- **`compare_candidate_stops(places=[...])`** — ranks 2–5 overnight
  candidates on weather + elevation + (later) services. One call
  instead of N×forecast calls → fewer chances to hit #19's loop
  detector.
- **`solar_charging_forecast(location)`** — cloud cover + sun hours
  for the next 1–3 days, rated for boondocking solar panels. Pure
  Open-Meteo data, just a different aggregation.
- **`find_rv_services_nearby(location, categories=[dump_station,
  potable_water, propane, laundry, grocery, truck_stop])`** —
  OpenStreetMap / Overpass API query. No key, fits the existing
  no-auth pattern. Categories map to OSM tags (`amenity=sanitary_dump_station`,
  `amenity=drinking_water`, etc.).
- **`get_crime_rate(location)`** — user-requested. FBI UCR data for
  US (free, bulk CSV, cached); for non-US punt to a "not available"
  response rather than guess. Useful companion to the overnight-stop
  decision.
- **`get_regional_news(region=None, category=None, limit=5)`** —
  headlines for a region. When `region` is omitted, default to the
  caller's country / state from `detect_my_location_by_ip` (same
  "defaults to 'here'" pattern as the weather shortcuts). Candidate
  free sources: GDELT Project (global events, no key), Google News
  RSS (per-country feeds, no key, rate-limited), mediastack /
  newsdata.io free tiers (keyed, stricter limits). Prefer GDELT +
  Google News RSS to stay key-free. Accuracy caveat in response
  guidance when the region came from GeoIP — same pattern as the
  weather "outside right now" shortcut. Response shape mirrors the
  radio-stations tool: `{"filters": {...}, "count": N, "articles":
  [{"title", "url", "source", "published_at", "snippet"}]}`.
- **`last_resort_web_search(query, max_results=5)`** — DuckDuckGo
  Instant Answer / HTML endpoint (no key, rate-limited but generous).
  General-purpose escape hatch for "I don't have a dedicated tool for
  this intent". Risk: small models over-use a generic `web_search`
  name and stop picking specific tools — so the name itself is part
  of the contract. Naming options to A/B on the eval harness before
  shipping: `web_search` (baseline, risks over-use), `fallback_web_search`
  (clear secondary signal), `last_resort_web_search` (strongest nudge
  but risks *under*-use — model refuses it even when it's the right
  call), `web_search_when_no_specific_tool_fits` (explicit but long,
  may hurt selection on other tools via token pressure). Docstring
  still spells out "prefer a specific tool when one exists" in every
  case; the naming experiment is about how much work the name itself
  does *before* the docstring even gets read.

Promotion rule: pick one, sketch the docstring, add 2–3 cases to
`cases.yaml`, run the eval matrix, file an issue only if the
numbers don't regress the existing families.
