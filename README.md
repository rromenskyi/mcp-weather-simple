# mcp-weather-simple

A simple [Model Context Protocol](https://modelcontextprotocol.io) server
that exposes weather tools to local LLMs (Ollama, Claude Desktop, Open
WebUI, etc.). Backed by [Open-Meteo](https://open-meteo.com/) — no API
key required.

Two transports in a single codebase:

- **stdio** — default, for local MCP clients that spawn the server as a
  subprocess (e.g. `mcphost`, Claude Desktop).
- **streamable HTTP** — for running the server as a container in
  Kubernetes and connecting remote MCP clients over the network.

## Tools

| Tool                                               | Description                                                                                                                                           |
|----------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------|
| `get_current_date(timezone="UTC")`                       | Today's date, weekday and the timezone used as anchor. Call this first when the user asks about "today" / "tomorrow" / a weekday. |
| `get_current_time_in_city(city, country_code=None)`      | Current local date, time, weekday, UTC offset and timezone for a city or postal code — answers "what time is it in Kyiv?" without arithmetic. |
| `detect_my_location_by_ip(ip=None)`                      | GeoIP lookup (ipwho.is, no key, HTTPS). Answers "where am I?" / "weather here?" by returning city, country, timezone, lat/lon and local clock. Feed `city` straight into the weather tools. Inside k8s this reports the cluster's egress IP city, not the browser's — pass `ip` explicitly when that matters. |
| `find_place_coordinates(city, country_code=None)`        | Resolve a city name or postal code to lat/lon and timezone. `country_code` (ISO-3166-1 alpha-2) disambiguates homonyms and zip-code collisions. |
| `search_places(query, country_code=None, feature_types=None, limit=5)` | Return every geocoding candidate for an ambiguous query — towns, mountains, lakes, islands, neighborhoods, airports, etc. Each candidate carries a `feature_type` human label; filter with `feature_types=["city"]` for only towns, or leave unset to see every kind. |
| `get_current_weather_in_city(city, country_code=None)`   | Current temperature, humidity, wind, conditions. |
| `get_weather_forecast(city, days=7, country_code=None)`  | Daily forecast, 1–16 days ahead. Each entry carries `day_label` ("today", "tomorrow", "in N days") anchored to the city's local timezone. |
| `get_hourly_forecast(city, hours=24, country_code=None)` | Hour-by-hour forecast (up to 168 h = 7 days). Timestamps in the city's local timezone. |
| `get_sunrise_sunset(city, date_iso=None, days=1, country_code=None)` | Sunrise, sunset, daylight duration and sunshine duration for one or more consecutive days. |
| `get_air_quality(city, country_code=None)`               | Current PM2.5 / PM10 / ozone / NO2 / SO2 / CO plus European and US AQI indices. |
| `get_weather_by_coordinates(latitude, longitude)`        | Current weather at raw lat/lon — skips the geocoder entirely. Useful with coordinates from `detect_my_location_by_ip` or pasted from a map app. |
| `get_historical_weather(city, start_date_iso, end_date_iso=None, country_code=None)` | Daily archive (1940–present). Up to 31 days per request. |
| `get_wikipedia_summary(title, lang="en")`                | ~300-char Wikipedia summary + page URL. Answers "tell me about X". |
| `get_country_info(country)`                              | Country facts: capital, population, currencies, languages, calling code, borders, timezones. Accepts ISO code or plain name. |
| `get_public_holidays(country_code, year=None)`           | Public / bank holidays for a country in a given year. Backed by `date.nager.at`. |
| `convert_currency(amount, from_currency, to_currency)`   | Fiat currency conversion at today's rate. ISO-4217 codes. |
| `list_radio_stations(country=None, tag=None, language=None, limit=10)` | Browse internet-radio stations by country, tag or language. Volunteer catalogue (radio-browser.info) with mirror fallback. |

### No-arg shortcuts for common questions

Pre-composed tools whose **name alone** tells the model which user question they answer. Each one auto-detects the caller's city via GeoIP (so the user doesn't have to name their city) and calls the appropriate lower-level tool.

| Tool | Answers the question |
|---|---|
| `get_weather_outside_right_now()` | "What's the weather outside?" / «какая погода на улице?» |
| `get_weather_forecast_for_today()` | "What's the weather today?" / «какая погода сегодня?» |
| `get_weather_forecast_for_tomorrow()` | "What's the forecast for tomorrow?" / «прогноз на завтра?» |
| `get_current_time_where_i_am()` | "What time is it?" / «который сейчас час?» |

The lower-level tools stay available for precise follow-ups (another city, a specific date, longer horizon).

## Multilingual queries

The geocoder auto-detects the query's writing system and picks the
matching Open-Meteo `language` index, with a fallback chain for
scripts that are shared across languages:

| Query example      | Detected script    | Language fallback chain |
|--------------------|--------------------|--------------------------|
| `Paris`, `90210`   | Latin              | `en`                     |
| `Москва`, `Одеса`  | Cyrillic (generic) | `ru` → `uk`              |
| `Київ`, `Львів`    | Cyrillic with Ukrainian-unique glyphs (`і ї є ґ`) | `uk` → `ru` |
| `Αθήνα`            | Greek              | `el`                     |
| `القاهرة`           | Arabic             | `ar`                     |
| `ירושלים`           | Hebrew             | `he`                     |
| `北京`, `東京`, `横浜` | CJK Han           | `zh` → `ja` → `ko`      |
| `東京タワー`        | Japanese (Kana)    | `ja`                     |
| `서울`             | Korean (Hangul)    | `ko`                     |

The helper tries each language in order and stops at the first
non-empty result set. Without this chain, `language=en` returns only
Latin-name hits — so a Cyrillic `Москва` used to resolve to two tiny
Tajik villages named `Moskva` instead of the Russian capital, and
`横浜` picked a non-existent Chinese hit instead of Yokohama-JP. The
fallback means callers can paste any name in its native script and
get the right city on the first try.

Postal-code queries (`90210`, `SW1A 1AA`) flow through the same
helper — US/DE/FR postal codes resolve cleanly, UK postcodes are not
indexed by Open-Meteo. Pair a postal code with `country_code=` to
avoid cross-country collisions (`10001` matches both New York, US
and Troyes, FR).

## Response envelope

Every successful tool response carries two top-level fields:

- `relay_to_user` (bool): `true` = the model can answer directly from
  this data; `false` = the model MUST clarify with the user before
  answering (ambiguous input, multiple candidates, duplicate call).
- `guidance` (str): plain-English one-liner telling the model what to
  do with the body — e.g. `"Relay directly."`, or `"Relay with a
  caveat: city was auto-detected from the caller's IP and may be
  wrong (VPN / NAT). If the user disagrees, ask for the city."` for
  GeoIP-backed shortcuts.

Small models follow short prose instructions better than they
interpret a controlled vocabulary (`confidence: enum`), so guidance
strings are kept under one sentence.

Tools that use GeoIP (`detect_my_location_by_ip`,
`get_weather_outside_right_now`, etc.) use a GeoIP-specific
`guidance` string that replaces the old `accuracy_warning` body
field — the uncertainty is now an instruction the LLM can't miss,
not an optional body hint. The `location_source` body field
(`"geoip_autodetected"` / `"geoip_explicit"`) is kept for
programmatic consumers.

## What is `uv`?

[`uv`](https://docs.astral.sh/uv/) is a fast Python project and
environment manager from Astral (authors of `ruff`). It replaces the
`pip` + `venv` + `pyenv` + `pip-tools` combo with a single Rust-based
tool.

In this project `uv`:

- reads `pyproject.toml` / `uv.lock` and creates a reproducible `.venv`;
- runs the server inside that venv without manual activation
  (`uv run server.py`);
- is the entrypoint used by `mcphost` to spawn the server on the host.

Install (user-local, no `sudo` required):

```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

The binary lands in `~/.local/bin/uv`.

First-time setup in this project:

```
uv sync          # creates .venv and installs mcp + httpx from pyproject.toml
uv run server.py # runs the server (stdio transport by default)
```

## Architecture

```
┌──────────┐       ┌─────────┐       ┌──────────────┐       ┌────────────┐
│   User   │──────▶│  HOST   │──────▶│    Ollama    │       │ MCP server │
│  (chat)  │       │ (bridge)│◀──────│   (in k3s)   │       │  weather   │
└──────────┘       └────┬────┘       └──────────────┘       └─────▲──────┘
                        │                                          │
                        └──────────────────────────────────────────┘
                              "call get_forecast('Moscow')"
```

Three moving parts:

- **Ollama** — the brain. Knows nothing about MCP. When given a list of
  tools it can reply "I want to call tool X with arguments Y".
- **MCP server** (this project) — exposes tools. Stateless, idle until
  called.
- **Host / bridge / MCP client** — the glue. Owns the chat loop, talks
  to Ollama over HTTP, and when the model asks for a tool it calls the
  MCP server and feeds the result back to Ollama.

Ollama running in k3s is just the right side of the diagram. The host
can live anywhere.

## Deployment options

The server supports two transports, switchable via the `MCP_TRANSPORT`
env var:

- `streamable-http` — used by the Docker image and Kubernetes manifests
  (Option A).
- `stdio` — default, used when an MCP client spawns the server as a
  subprocess on the same host (Option B).

The HTTP transport exposes three unauthenticated probe paths, split
by Kubernetes convention. All three bypass the MCP initialize
handshake and the bearer-auth middleware so probes are cheap and
never need a token.

- **`/healthz` and `/livez`** — liveness. No I/O; always 200 OK with
  `{"status": "ok", "service": "mcp-weather", "probe": "liveness"}`.
  A hung upstream dependency cannot flip liveness, so k8s will never
  restart the pod just because Open-Meteo is slow.
- **`/readyz`** — readiness. Actively probes Open-Meteo's geocoder
  with a 2 s timeout and reports the result. On success: 200 OK with
  `{"status": "ok", "probe": "readiness", "checks": {"open_meteo": "ok"}}`.
  On upstream failure: 503 with `{"status": "not_ready", ...,
  "checks": {"open_meteo": "ReadTimeout"}}` — k8s then drains
  traffic until upstream clears instead of our server faking
  readiness.

```
curl -sf http://127.0.0.1:8000/healthz
curl -sf http://127.0.0.1:8000/readyz
```

### Option A — everything in k3s (production path)

#### 1. Build and push the image

```
docker build -t ghcr.io/<user>/mcp-weather-simple:0.1.0 .
docker push     ghcr.io/<user>/mcp-weather-simple:0.1.0
```

Local smoke test (the HTTP endpoint is `/mcp`):

```
docker run --rm -p 8000:8000 ghcr.io/<user>/mcp-weather-simple:0.1.0
curl -X POST http://127.0.0.1:8000/mcp \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json, text/event-stream' \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize",
       "params":{"protocolVersion":"2025-03-26","capabilities":{},
                 "clientInfo":{"name":"curl","version":"0"}}}'
```

You should get an SSE frame back with `serverInfo.name = "weather"`.

#### 2. Deploy to the cluster

Manifests live in `k8s/` (Kustomize). Edit the namespace and image to
fit your cluster — the defaults are `ai` and
`ghcr.io/rromenskyi/mcp-weather-simple:latest`.

```
kubectl apply -k k8s/
```

Inside the cluster the server becomes reachable at
`http://mcp-weather.ai.svc:8000/mcp`.

If your MCP client (Open WebUI / mcphost) also runs in the cluster you
don't need external ingress. To expose the server outside the cluster,
uncomment `ingressroute.yaml` in `kustomization.yaml` and set your
hostname. **When you do, also relax the NetworkPolicy** — the default
`k8s/networkpolicy.yaml` only lets the `ai` namespace in, so Traefik
running in its own namespace (`traefik-system`, `ingress-controller`,
…) will get its traffic dropped at the CNI layer. A ready-to-use
cross-namespace rule is commented out at the bottom of the
NetworkPolicy; uncomment it and point the `namespaceSelector` at your
Traefik install.

The manifests are compatible with the PodSecurity `restricted` profile:
`runAsNonRoot`, `readOnlyRootFilesystem`, `capabilities.drop: [ALL]`,
`seccompProfile: RuntimeDefault`, and bounded CPU/memory.

#### 3. Create the auth Secret

The server enforces Bearer-token auth on the HTTP transport when
`MCP_AUTH_TOKEN` is set (the Deployment wires it to a Secret named
`mcp-weather-auth`). Generate a token and create the Secret:

```
TOKEN=$(openssl rand -hex 32)
kubectl -n ai create secret generic mcp-weather-auth \
  --from-literal=token="$TOKEN"
echo "$TOKEN"   # share with the MCP client
```

Requests without `Authorization: Bearer $TOKEN` get a `401`.

The `k8s/networkpolicy.yaml` manifest restricts inbound traffic to pods
in the same namespace (`ai`). Combined with the Bearer token this
gives two independent layers: the CNI blocks the packet and the server
rejects the request. Egress is locked down to DNS + HTTPS to public
IPs (Open-Meteo).

If you don't want auth (e.g. purely local testing), leave
`MCP_AUTH_TOKEN` unset — the middleware will not be attached.

#### 4. Wire it to Ollama through a chat UI

Deploy a chat frontend with MCP support next to Ollama. **Open WebUI**
is the common choice (Helm chart `open-webui/open-webui`). In its
settings, point it at:

- Ollama endpoint → `http://ollama.<ns>.svc:11434`
- MCP server → `http://mcp-weather.ai.svc:8000/mcp` with header
  `Authorization: Bearer <TOKEN>`

Result: three pods in the cluster (Ollama + Open WebUI + mcp-weather),
the model calls weather tools on its own, the user talks to it in a
browser.

### Option B — host on your laptop (dev / fast start)

Useful when you want to experiment without touching the cluster.

1. Forward Ollama out of the cluster:

   ```
   kubectl -n <namespace> port-forward svc/ollama 11434:11434
   ```

2. Install [`mcphost`](https://github.com/mark3labs/mcphost) — the
   Ollama ↔ MCP bridge:

   ```
   go install github.com/mark3labs/mcphost@latest
   # or grab a release binary from the GitHub releases page
   ```

3. Run it:

   ```
   mcphost --config /path/to/mcp-weather-simple/mcphost.config.json \
           -m ollama:llama3.1
   ```

   Use a model that supports tool calling: `llama3.1`, `qwen2.5`,
   `mistral-nemo`, `llama3.2`, etc. Models like `phi` or `gemma2` won't
   work.

The included `mcphost.config.json` spawns the server via `uv` with
stdio transport (no auth, no container) — nothing else to configure.

#### Debug the server without Ollama

```
uv run mcp dev server.py
```

This launches the MCP Inspector in a browser so you can call tools by
hand and inspect the traffic.

## Testing

Two layers, each answering a different question.

### Unit tests — does the server behave?

`tests/test_server.py` (35+ cases, ~1 s total, respx-mocked — no
network). Proves every tool returns the correct response shape for
a known input: correct geocoding fallback across scripts, correct
`day_label` anchoring to the city's timezone, correct per-hour /
per-day row structure, Open-Meteo parameters wired right, friendly
error messages for 4xx vs 5xx, lat/lon validation, mirror fallback
for radio-browser, etc.

Run: `uv sync --extra test && uv run pytest -q`

Every PR runs these automatically via `.github/workflows/build.yml`.
Think of it as "the plumbing works, input X produces output Y".

### Tool-calling eval — do LLMs actually pick the right tool?

Separate problem: even a perfectly-implemented tool is useless if
the model ignores it. Small open-source LLMs (qwen2.5:7b,
qwen2.5:14b, llama3.1:8b, …) read the tool's **description** and
name, and decide on the fly whether to invoke it. A docstring that
reads clearly to a human but overlaps semantically with a sibling
tool will cause the model to fumble the choice.

`tests/integration/eval_tool_calling.py` + `tests/integration/cases.yaml`
measure this as a **top-1 tool-selection hit rate**:

- 40+ `(prompt, expected_tool)` pairs grouped by intent family —
  weather-here, forecast-tomorrow, hourly, air quality, sunrise,
  historical, time-in-city, time-here, currency, country-info,
  public holidays, radio stations, Wikipedia, places-disambiguation.
- Russian + English mixed on purpose: the geocoder has a
  script-fallback chain, and the model has to trigger tools the
  same way regardless of query language.
- For every case the harness sends `{prompt, tools, temperature=0,
  seed=42}` to Ollama's `/api/chat`, reads `tool_calls[0].function.name`,
  scores it against the expected tool.
- Reports a **per-family** and **overall** hit rate, lists the
  failing prompts. `pass_threshold` lives in `cases.yaml`; current
  baseline on qwen2.5:7b is **~97 %** (86.8 % on the first run
  before a warm-up call was added to neutralise the cold-start
  ReadTimeout on the first 4 scored cases).

Run (GitHub Actions): `.github/workflows/eval.yml` kicks this off
via `workflow_dispatch` and a nightly `cron` at 03:00 UTC. Each
runner gets 16 GB RAM / 4 vCPU, its own Ollama model cache
(`ollama-<model>-v1`), and its own job summary with the scored table.

Three matrix axes, fan-out independently:

- **model** — `qwen2.5:7b` + `qwen2.5:14b` by default (leave input
  empty), or pick a single model via the `model` dropdown.
- **output_schema** — `off` (default), `on`, or `both`. The `both`
  setting runs the outputSchema A/B experiment; see ROADMAP's
  "Experiments → outputSchema A/B" section.
- **chunk_count** — `1`, `2` (default), or `4`. Shards the 40-prompt
  suite across parallel rows so the 14b-on-CPU path fits inside the
  45-minute per-row ceiling. Widest fanout (2 × 2 × 4 = 16 rows) is
  still under the 20-concurrent-job free-tier cap.

**Named recipes**: `scripts/eval.sh <recipe>` wraps `gh workflow run`
so you don't have to remember axis combinations. Recipes: `quick`
(single 7b, 1 chunk — ~15 min), `matrix` (default nightly shape),
`schema-ab` (A/B experiment, 8 parallel rows), `full` (widest fanout,
16 rows), `14b-only`. Run `scripts/eval.sh help` for the full list.

Step logs stream live — each per-case line appears as it happens via
`PYTHONUNBUFFERED=1` + `python -u` + `flush=True`, so you can watch
progress instead of waiting for a final wall of text.

Run (locally, against your own Ollama): point the script at a
running Ollama and MCP server:

```bash
OLLAMA_URL=http://127.0.0.1:11434 \
MCP_URL=http://127.0.0.1:8000/mcp \
OLLAMA_MODEL=qwen2.5:7b \
uv run python tests/integration/eval_tool_calling.py
```

**Why two layers:** the unit suite tells you *the weather endpoint
works*, the eval tells you *the chatbot would actually call it when
the user says «какая погода»*. A failing eval hit rate on a single
family (say `get_wikipedia_summary` drops from 2/2 to 0/2) is the
clearest signal that that tool's docstring needs tuning — there's
no other place to point when "it worked in curl but the model won't
use it" bites.

## Project files

- `server.py` — the MCP server (FastMCP). Transport is controlled by
  `MCP_TRANSPORT` (`stdio` | `streamable-http`), port by `MCP_PORT`.
- `pyproject.toml` / `uv.lock` — dependencies (`mcp[cli]`, `httpx`).
- `mcphost.config.json` — `mcphost` config for Option B.
- `tests/` — unit tests (`test_server.py`, mocked) and integration
  eval harness (`integration/`, hits real Ollama + MCP).
- `Dockerfile` / `.dockerignore` — multi-stage image built with `uv`,
  runs as non-root with a read-only root filesystem.
- `k8s/` — Kustomize manifests: Deployment, Service, NetworkPolicy,
  Secret template, and an optional Traefik IngressRoute.
- `.github/workflows/` — `build.yml` (image + python import on every
  PR), `eval.yml` (tool-calling eval, manual + nightly).

## License

MIT.
