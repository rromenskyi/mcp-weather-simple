# Tool catalog scaling — when & how to split

Working notes on what to do when the 23-tool catalog starts to hurt.
Written 2026-04-22, before a split is needed — captured so we don't
re-derive it from scratch next time.

## Where we are today

- **23 tools** across 5 domains (weather, location, currency, radio,
  wikipedia), served from one MCP server.
- Tool catalog ≈ **4500 tokens** (roughly 3000 docstring + 1400 JSON
  schema, measured when we bumped `OLLAMA_CONTEXT_LENGTH` to 8192).
- That's **~55 %** of qwen3.5:9b's working context on every request.
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

### 3. MCP `notifications/tools/list_changed` (spec-correct, client-gated)

The protocol-native way. Session starts with a single `set_context(domain)`
tool exposed. Model calls it. Server updates session state, fires
`tools/list_changed`. A compliant client re-fetches `tools/list` and
now sees the narrow per-domain subset.

Three real catches:

- **Client support is uneven.** Spec recommends re-fetching, but
  mcphost historically loads the catalog once at startup. Open WebUI
  0.8.x — unverified; given their existing tool-forwarding bug
  (upstream issue #23907), don't assume it works.
- **First-turn penalty.** Every conversation starts with an extra
  round-trip for the routing call. On CPU-only that's +5-10 s; on L4
  it's negligible.
- **Cross-domain queries need multi-domain mode.** `set_context` ends
  up taking a list, and the model is back to "pick the right set",
  which is most of what the routing call was meant to avoid.

Worth doing only *after* verifying the target clients actually honor
the notification. Quick experiment: stand up a FastMCP with 2 initial
tools and 5 additional ones exposed via `list_changed` after the
first call. If mcphost / OWUI then invoke the additional tools, the
mechanism is live.

### 4. Router-agent (two-pass LLM)

First LLM call has no tools, returns the domain label. Second LLM
call uses the domain's narrow catalog. Guaranteed to work, no
protocol tricks. Cost: 2× latency, ~2× tokens billed (for remote
APIs; for local Ollama it's just wall-clock).

Reach for this only if we end up behind an API where latency is
already a non-issue or we specifically want the routing call to
produce structured output we log for analytics.

### 5. Meta-tool consolidation (avoid)

> Replace `get_weather_current` / `get_weather_daily` / `get_weather_hourly`
> with one `weather(action: Literal[...], ...)` tool.

Catalog collapses dramatically — on paper. In practice small models
do measurably *worse* at enum dispatch inside a single tool than at
picking from distinct tool names. The tool-picking attention head
seems to be a different (and sharper) mechanism than generic
argument-filling. Mentioned here so we don't re-invent it when the
idea resurfaces.

## Recommendation while we're still in the green

Keep the monolith. Culturally treat every new tool as paying for
itself — it should either (a) hit a use case no existing tool covers
or (b) materially bump eval hit rate. Don't add utility tools because
"nice to have". The first real split happens when a deployment
appears whose scope is genuinely narrow (option 1), not as a
preemptive cleanup.

If we cross the signal thresholds before that (90 % hit rate or 6K
prefill), the order of moves is:

1. Prune obvious redundancy (merge near-synonym tools, drop anything
   the eval doesn't exercise).
2. Option 2 (first-message heuristic) — cheap, client-agnostic.
3. Option 1 (deployment split) — when a narrow-scope deployment
   actually materializes.
4. Option 3 (`list_changed`) — only after confirming the target
   clients honor it.
