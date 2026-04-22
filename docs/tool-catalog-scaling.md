# Tool catalog scaling — when & how to split

Originally written 2026-04-22 as design theory; **updated later the
same day with measured outcomes from the experiments below**. The
theoretical sections are kept because the reasoning still applies
next time we're making a decision here; see the [Experiment results](#experiment-results-2026-04-22)
block for what actually shipped.

## Where we are today

- **23 tools** across 5 domains (weather, location, currency, radio,
  wikipedia), served from one MCP server.
- Default catalog (`MCP_ROUTER_MODE=off`) ≈ **5289 tokens**
  (21,157 chars of tool-list JSON @ ~4 chars/token).
- That's **~64 %** of qwen3.5:9b's 8K working context on every
  request.
- Each new tool adds **~150-300 tokens** (name + docstring + schema).

Two independent costs grow with catalog size:

1. **Prefill cost — linear.** Every request pays the full catalog on
   prefill. On CPU-only GHA runners this is already the dominant
   cost; on L4 it's <5 s and irrelevant.
2. **Hit rate — step function, not linear.** Small OSS models
   (≤ 10 B) hold ~10-15 semantically-distinct tools reliably. Past
   that, near-synonym tools start competing (`get_weather_by_coords`
   vs `get_hourly_forecast` is the canonical example). Anthropic's
   guideline is ≤ 30 tools *if they don't overlap*; ours span 5
   domains in one server, which is already on the edge.

Signal that "it's time to split": eval top-1 hit rate on 9b drops
below 90 %, or prefill crosses 6K tokens. We have roughly 5 more
features of headroom.

## Experiment results (2026-04-22)

Three router variants built, gated behind `MCP_ROUTER_MODE`
(**default flipped to `fat_tools_lean` in the same 2026-04-22 round
once the numbers below landed**). Measured on qwen3.5:9b via eval
harness (`tests/integration/eval_tool_calling.py`) running against
the L4 self-hosted runner — same 44 cases each.

| Mode | Tools visible | Catalog tokens | vs off | Hit rate | Δ hit rate |
|---|---:|---:|---:|---:|---:|
| `off` (monolith)     | 23 | 5289 | —    | 41/44 = 93.2 % | baseline |
| `list_changed`       | dynamic | dynamic | n/a  | **DNF** | clients ignore the notification |
| `fat_tools`          |  4 | 1885 | **-64 %** | 41/44 = 93.2 % | **0** |
| `fat_tools_lean` *(default)* | 4 | **1090** | **-79 %** | 41/44 = **93.2 %** | **0** |

Three identical 93.2 %'s are the **same 3 failures** in every mode:
"What's the capital of Ukraine?" / "столица Нидерландов?" /
"Which Springfield is the user asking about?" — the model answers
from memory instead of tool-calling, arguably the right call. Not
a router artefact.

**Bottom line: ~4200 tokens of prefill bought back (−79 % of
catalog) for ~100 LOC of `fat_tools_lean.py` + a module-level
constant, with zero hit-rate regression.** On a 2-vCPU CPU at
~15 tok/s prefill, that's **4-5 minutes saved on every first-turn
request**; on GPU it's single-digit seconds but still free. Server
default is now `fat_tools_lean`; set `MCP_ROUTER_MODE=off` to pin
the historical monolith surface (nightly eval cron does this so
baseline trend data stays comparable, and it's the escape hatch if
a specific client turns out to misbehave on the fat surface).

Conclusions that flipped during the experiments:

- **`list_changed` is spec-correct but dead in our stack.**
  Confirmed empirically on the L4 runner. mcphost reads `tools/list`
  once at connection (see `mark3labs/mcphost internal/tools/mcp.go
  loadServerTools`) and never refetches; Open WebUI has zero
  references to `list_changed` anywhere in its codebase. A compliant
  server sending the notification accomplishes nothing — the client
  stays locked to the original catalog. PR #48 shipped the prototype
  behind `MCP_ROUTER_MODE=list_changed`; kept around as a
  reference implementation and a null-result data point, not a
  deployment path.
- **The "meta-tool consolidation regresses small models" claim
  (section 5 below, original wording) was wrong for our shape.**
  On qwen3.5:9b, a fat `weather(action, city, ...)` tool picks
  identically well to distinct `get_current_weather_in_city` /
  `get_hourly_forecast` / ... In other words the tool-picking
  attention head doesn't seem to penalise action-enum dispatch
  when the docstring is clear. Possibly a qwen3-family specific
  observation — worth re-testing if we add a smaller model.
- **Most of `fat_tools`' 1885 tokens is FastMCP-generated noise.**
  Every optional `str | None` kwarg expands to
  `anyOf: [{type: string}, {type: null}]` + a `title`. `fat_tools_lean`
  collapses the signature to `(action, params: dict = {})` — drops
  all the shell, model learns param shapes from the docstring.
  Another **42 %** off the already-shrunk `fat_tools` catalog, with
  hit rate still TBD (pending the lean A/B run).

Recommended default going forward (revised):

- **`off` on L4 / GPU** — prefill is <5 s anyway, the extra schema is
  free, stay with the tool-picking surface the model does best on.
- **`fat_tools_lean` on i7 / CPU** — prefill on the 23-tool monolith
  takes 3-7 min on CPU per first-turn; the lean variant drops that
  to ~45-90 s. When the lean A/B lands confirming hit rate is fine,
  flip platform sidecars to this by default.

The experiments also produced two reusable primitives:

- `fat_tools_map.py` holds `NARROW_TO_FAT: {narrow_name → (fat, action)}`.
  Both the router modes AND the eval scorer read from it, so
  cases.yaml scores against any of the three surfaces without
  branching. Drift-detected by
  `test_narrow_to_fat_map_covers_every_registered_tool`.
- `eval_tool_calling.py` canonicalises `expected` and `picked` tool
  names to a `fat(action)` form when `MCP_ROUTER_MODE` is either
  `fat_tools` or `fat_tools_lean`. Same scorer code path, same
  thresholds.

## The obvious idea that doesn't work

> Split into 3 MCP servers (weather / location / knowledge) and the
> catalog shrinks automatically.

**No, it doesn't.** mcphost and Open WebUI load every configured MCP
server at startup and send the *union* of their tool lists to the
model on every request. Three servers in the config = same 23 tools
in the prompt. Splitting gives you **deployment flexibility** (one
config picks only weather), not *in-session* dynamic narrowing.

## Real options for narrowing the catalog

Ordered roughly by implementation cost.

### 1. Per-deployment filtering (do this first)

Different `mcphost.config.json` files / Open WebUI tenants wire only
the MCP servers they actually need. A voice-weather assistant gets
only `mcp-weather`; the full chat gets everything. Zero protocol
complexity. Not dynamic-per-query — dynamic-per-deployment.

Natural split points on the current codebase:

- `mcp-weather` — get_current, forecast_daily, forecast_hourly,
  air_quality, sunrise_sunset, historical (≈ 8 tools, ~1500 tokens).
- `mcp-location` — geocoding, address parse/normalize, ip lookup,
  detect location (≈ 5 tools, ~900 tokens).
- `mcp-knowledge` — wikipedia, currency, holidays, radio
  (≈ 10 tools, ~2100 tokens).

Do this when we get a deployment whose scope genuinely doesn't need
the whole catalog. Not before.

### 2. Server-side first-message heuristic (cheap, works today)

The server looks at the incoming session's system prompt / first user
message at `tools/list` time and returns a subset based on keywords.
No client changes, no MCP-protocol extensions, works with every
client including the ones that don't honor `list_changed`.

```python
@mcp.list_tools()
async def list_tools(session: Session):
    hint = _classify_session(session)  # "weather" | "knowledge" | None
    return ALL_TOOLS if hint is None else DOMAIN_TOOLS[hint]
```

Downsides: heuristic is grubby (keyword matching is language-aware
the wrong way, misses paraphrase), and first-message classification
is fragile in multi-turn chats where the topic shifts. Cross-domain
queries ("100 USD in UAH and weather in Kyiv") degrade silently.

### 3. MCP `notifications/tools/list_changed` (spec-correct, ❌ dead in our stack)

The protocol-native way. Session starts with a single `set_context(domain)`
tool exposed. Model calls it. Server updates session state, fires
`tools/list_changed`. A compliant client re-fetches `tools/list` and
now sees the narrow per-domain subset.

**Shipped as PR #48 for the express purpose of verifying this
empirically. Clients don't honour the notification:**

- **mcphost:** `internal/tools/mcp.go` calls `client.ListTools` once
  inside `loadServerTools` at connection start. No handler for
  `notifications/tools/list_changed`, no refetch logic. Cache is
  frozen for the entire session.
- **Open WebUI:** 0 matches for `list_changed` anywhere in the
  `open-webui/open-webui` codebase (GitHub code-search, 2026-04-22).
- **Claude Desktop:** likely the one compliant client, per Speakeasy's
  dynamic-tool-discovery writeup. Not in our stack.

The prototype is still in main behind `MCP_ROUTER_MODE=list_changed`
as a reference implementation — documents the mechanism if a future
client lands support. Don't expect it to do useful work in the
foreseeable future.

(Historical theoretical downsides — first-turn latency penalty, the
cross-domain query degradation — were moot once we confirmed client
support is absent.)

### 4. Router-agent (two-pass LLM)

First LLM call has no tools, returns the domain label. Second LLM
call uses the domain's narrow catalog. Guaranteed to work, no
protocol tricks. Cost: 2× latency, ~2× tokens billed (for remote
APIs; for local Ollama it's just wall-clock).

Reach for this only if we end up behind an API where latency is
already a non-issue or we specifically want the routing call to
produce structured output we log for analytics.

### 5. Meta-tool consolidation ("fat tools") — ✅ shipped, works

> Replace `get_weather_current` / `get_weather_daily` / `get_weather_hourly`
> with one `weather(action: Literal[...], ...)` tool.

Catalog collapses dramatically — measured −64 % vs monolith (5289 →
1885 tokens), with **zero hit-rate regression** on qwen3.5:9b (93.2 %
either way, same 3 misses on the same cases). The doc's original
theoretical concern ("small models do worse at enum dispatch") did
not show up on this model; re-test if we add something smaller than
9 B.

Variants implemented and gated behind `MCP_ROUTER_MODE`:

- **`fat_tools`** (PR #50) — 4 domain-tools with per-action named
  kwargs. 1885 tokens.
- **`fat_tools_lean`** (PR #57) — same 4 tools with
  `(action, params: dict)` signature; drops FastMCP's nullability
  shell (`anyOf:[{str},{null}]` per optional kwarg). 1090 tokens.

See `fat_tools.py`, `fat_tools_lean.py`, `fat_tools_map.py`.

## Recommendation (2026-04-22, post-experiments)

**GPU-backed deployments (L4 / future home GPU):** stay on
`MCP_ROUTER_MODE=off`. Prefill is <5 s at 5k tokens; the schema
noise is effectively free and keeps the model on its most robust
tool-picking surface.

**CPU-only deployments (platform on i7, and anything without GPU):**
flip to `MCP_ROUTER_MODE=fat_tools_lean` (once the lean A/B run
confirms hit rate holds). That's the deciding-factor variable — if
a deployment runs chat on CPU, the 3-7 min first-turn prefill on
the monolith is the single biggest UX tax this codebase has, and
the lean variant kills 80 % of it for free.

**Adding new tools:** each one should either cover a genuinely
missing use case or bump eval hit rate. Don't add "nice-to-have"
utilities. Growth is cheap in `fat_tools_lean` mode (~30-60 tokens
for an extra action on an existing fat tool) but still costs at the
margin.

### If we ever need dynamic narrowing

- `list_changed` is *not* a path — confirmed dead in our client
  stack. Reconsider only if mcphost / Open WebUI explicitly ship
  support.
- Option 2 (first-message heuristic) remains feasible and
  client-agnostic; the knob to reach for if static catalog still
  hurts after fat_tools_lean is live.
- Option 1 (deployment split) — when a clearly narrow scope
  materialises (e.g. a voice-weather skill that only needs the
  weather subset).
