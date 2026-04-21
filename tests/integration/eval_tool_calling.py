"""Tool-calling eval harness.

Runs every prompt from `cases.yaml` against a live Ollama model that has
the MCP tool catalog attached. Scores top-1 tool selection accuracy
overall and per intent family, prints a table, and exits non-zero when
the hit rate dips below `pass_threshold`.

This is an **integration** test, not a unit test — it needs:
  - a running Ollama (`OLLAMA_HOST`, default http://127.0.0.1:11434)
  - a running MCP server (streamable HTTP on `MCP_URL`, default
    http://127.0.0.1:8000/mcp) whose `tools/list` gets bundled into the
    Ollama request as native `tools=[...]`.

GitHub Actions runs both in the same job (see .github/workflows/eval.yml).
Locally, you can point it at already-running services from `./tf` or
a docker-compose. Either way the script is inert when MCP or Ollama
isn't reachable — it prints a skip-reason and exits 0.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import httpx
import yaml


# MCP streamable-HTTP — one round trip per tool request. We reuse a
# single httpx.AsyncClient across the whole eval so keep-alive kicks in
# and re-sessions between cases don't drown in TLS / DNS overhead.
class MCPClient:
    """Minimal MCP streamable-HTTP client. Only `initialize` + `tools/list`.

    We don't call `tools/call` here — the eval scores *which* tool the
    model picks, not whether it runs correctly. Tool invocation is
    covered by the unit-test suite (respx-mocked) and by live traffic
    in production.
    """

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url
        self._client = httpx.AsyncClient(timeout=10.0)
        self._session_id: str | None = None

    async def __aenter__(self) -> "MCPClient":
        return self

    async def __aexit__(self, *exc) -> None:
        await self._client.aclose()

    async def _post(self, payload: dict) -> dict:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id
        r = await self._client.post(self.base_url, json=payload, headers=headers)
        r.raise_for_status()
        # Streamable-HTTP wraps the JSON in an SSE envelope; strip it.
        text = r.text
        sid = r.headers.get("mcp-session-id") or r.headers.get("Mcp-Session-Id")
        if sid:
            self._session_id = sid
        m = re.search(r"data: ({.+})", text)
        return json.loads(m.group(1)) if m else json.loads(text)

    async def tools_list(self) -> list[dict]:
        await self._post({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "eval-harness", "version": "0"},
            },
        })
        resp = await self._post({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        return resp.get("result", {}).get("tools", []) or []


def mcp_tools_to_ollama(tools: list[dict]) -> list[dict]:
    """Convert MCP tool specs to Ollama's native `tools=` array shape.

    MCP returns:   {name, description, inputSchema}
    Ollama wants:  {type: "function", function: {name, description, parameters}}
    """
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("inputSchema", {"type": "object", "properties": {}}),
            },
        }
        for t in tools
    ]


async def run_case(
    client: httpx.AsyncClient,
    ollama_url: str,
    model: str,
    prompt: str,
    tools: list[dict],
) -> tuple[str | None, dict | None, float]:
    """Send one prompt to Ollama, return (tool_name | None, arguments | None, latency_s).

    `arguments` is the dict the model passed to the picked tool. We
    surface it alongside the name so the harness can diagnose the
    "right tool + wrong args" failure mode — e.g. `search_places`
    called with `query="Bountiful, Utah, 84010"` (correct tool, broken
    argument that the single-token contract from #14 is designed to
    prevent). Scoring is still name-only; arguments are informational.
    """
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "tools": tools,
        "stream": False,
        # Deterministic: minimum temperature + fixed seed so the eval is
        # reproducible across CI runs. Otherwise consecutive runs on the
        # same model can disagree on marginal cases.
        "options": {"temperature": 0, "seed": 42},
    }
    t0 = time.monotonic()
    r = await client.post(f"{ollama_url}/api/chat", json=payload)
    latency = time.monotonic() - t0
    r.raise_for_status()
    data = r.json()
    calls = (data.get("message") or {}).get("tool_calls") or []
    if not calls:
        return None, None, latency
    fn = calls[0].get("function") or {}
    return fn.get("name"), fn.get("arguments"), latency


def _format_arguments(arguments: dict | None) -> str:
    """Render a tool-call's arguments as `key=value, key=value` for the log.

    Stays compact — the eval log already includes the prompt and the
    picked tool, so this is the third column, not a JSON blob. None /
    empty renders as an empty string so the display stays tidy for the
    "no tool_call" rows.
    """
    if not arguments:
        return ""
    parts = []
    for k, v in arguments.items():
        if isinstance(v, str):
            parts.append(f'{k}="{v}"')
        else:
            parts.append(f"{k}={v}")
    return "(" + ", ".join(parts) + ")"


def _latency_stats(rows: list[dict]) -> tuple[float, float, float]:
    """(mean, p50, p95) in seconds over rows with non-zero latency.

    Zero latencies come from cases that errored before the HTTP call
    completed (see the `latency=0.0` assignment in `main_async`'s
    exception branch), so including them would skew the aggregate
    downward. A family where every case errors shows up as `—` in the
    table.
    """
    lats = sorted(r["latency"] for r in rows if r["latency"] > 0)
    if not lats:
        return 0.0, 0.0, 0.0
    n = len(lats)
    mean = sum(lats) / n
    p50 = lats[n // 2]
    # NIST-style p95: cap at last index so short families don't wrap.
    p95 = lats[min(n - 1, max(0, int(round(n * 0.95)) - 1))]
    return mean, p50, p95


def _fmt_latency(stats: tuple[float, float, float]) -> str:
    mean, p50, p95 = stats
    if mean == 0:
        return "       —"
    return f"{mean:5.1f}s / {p50:5.1f}s / {p95:5.1f}s"


def summarise(results: list[dict], threshold: float) -> int:
    """Print a pass/fail + latency table grouped by intent family,
    return exit code."""
    total = len(results)
    hits = sum(1 for r in results if r["picked"] == r["expected"])
    rate = hits / total if total else 0.0

    # Family = common prefix of the expected tool (e.g. `get_weather_*`).
    by_family: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        family = "_".join(r["expected"].split("_")[:2])  # get_weather, get_current, list_radio, …
        by_family[family].append(r)

    # Header includes latency (mean / p50 / p95) — lets the operator
    # spot cases where an outputSchema or other tool-catalog change
    # shifts inference speed even when hit rate looks flat.
    print()
    header = (
        f"{'family':<28} {'hits':>6} / {'total':>5}  {'rate':>6}   "
        f"{'mean / p50 / p95 latency':>28}"
    )
    print(header)
    print("-" * len(header))
    for family, rows in sorted(by_family.items()):
        fhits = sum(1 for r in rows if r["picked"] == r["expected"])
        ftotal = len(rows)
        print(
            f"{family:<28} {fhits:>6} / {ftotal:>5}  {fhits / ftotal:>6.1%}   "
            f"{_fmt_latency(_latency_stats(rows)):>28}"
        )
    print("-" * len(header))
    print(
        f"{'OVERALL':<28} {hits:>6} / {total:>5}  {rate:>6.1%}   "
        f"{_fmt_latency(_latency_stats(results)):>28}   (threshold {threshold:.0%})"
    )

    print()
    failures = [r for r in results if r["picked"] != r["expected"]]
    if failures:
        print("Failures:")
        for f in failures:
            picked = f["picked"] or "<no tool_call>"
            args_str = _format_arguments(f.get("arguments"))
            # Arguments live on a continuation line so long JSON doesn't
            # wreck the alignment of the main failures column.
            print(f"  × {f['prompt'][:60]:<62}  expected={f['expected']:<36} picked={picked}")
            if args_str:
                print(f"      args: {args_str}")

    if rate < threshold:
        print(f"\nFAIL: {rate:.1%} < threshold {threshold:.0%}")
        return 1
    print(f"\nPASS: {rate:.1%} ≥ threshold {threshold:.0%}")
    return 0


async def main_async(args: argparse.Namespace) -> int:
    cases_path = Path(args.cases)
    cfg = yaml.safe_load(cases_path.read_text())
    all_cases = cfg["cases"]
    cases = _slice_cases(all_cases, args.chunk_index, args.chunk_count)
    threshold = cfg.get("pass_threshold", 0.70)
    if args.chunk_count > 1:
        print(
            f"Sharded: {len(cases)}/{len(all_cases)} cases "
            f"(chunk {args.chunk_index + 1}/{args.chunk_count})"
        )

    # Fetch the tool catalog once — it doesn't change between cases.
    try:
        async with MCPClient(args.mcp_url) as mcp:
            tools = await mcp.tools_list()
    except Exception as e:
        print(f"SKIP: MCP unreachable at {args.mcp_url} ({type(e).__name__}: {e})")
        return 0
    if not tools:
        print(f"SKIP: MCP returned an empty tool list from {args.mcp_url}")
        return 0
    print(f"Loaded {len(tools)} tools from {args.mcp_url}")
    ollama_tools = mcp_tools_to_ollama(tools)

    # Probe Ollama before running the whole suite — no point burning
    # 10 min of CI only to find the model wasn't pulled.
    async with httpx.AsyncClient(timeout=args.timeout) as client:
        try:
            probe = await client.get(f"{args.ollama_url}/api/tags")
            probe.raise_for_status()
        except Exception as e:
            print(f"SKIP: Ollama unreachable at {args.ollama_url} ({type(e).__name__}: {e})")
            return 0
        installed = [m["name"] for m in probe.json().get("models", [])]
        if args.model not in installed:
            print(f"SKIP: model {args.model!r} not installed on this Ollama. Have: {installed}")
            return 0

        # Warm-up: two separate concerns, both need to be hot before the
        # first SCORED case or the 1st/2nd scored calls eat their entire
        # per-request timeout:
        #
        # 1. Model weights loaded into RAM / VRAM. Cheap (~5 s on CPU,
        #    ~1 s on GPU) once the weights are on disk.
        # 2. KV cache for the ~5 K-token tool catalog warm. Without it,
        #    the first scored case pays a full cold-prefill tax:
        #    ~250 s on 2-vCPU × 14b, which blew past even the 480 s
        #    timeout on 2026-04-21 — cases 1 and 2 both <error:
        #    ReadTimeout>, case 3 succeeded in 228 s once the prefix
        #    cache was warm.
        #
        # Fix: warm-up sends the IDENTICAL payload shape the scored
        # loop will use (`tools=[...]`, `temperature=0`, `seed=42`) so
        # Ollama's prefix caching actually primes the catalog. We keep
        # `num_predict=4` so generation stays short and the full call
        # fits inside the scored per-request timeout. Discard the
        # result — we only care about the warm KV cache side-effect.
        print("Warming up model (weights + tool-catalog KV cache)...", end=" ", flush=True)
        t0 = time.monotonic()
        try:
            warm_resp = await client.post(
                f"{args.ollama_url}/api/chat",
                json={
                    "model": args.model,
                    "messages": [{"role": "user", "content": "warmup — answer 'ok'"}],
                    "tools": ollama_tools,
                    "stream": False,
                    "options": {"temperature": 0, "seed": 42, "num_predict": 4},
                },
            )
            warm_resp.raise_for_status()
            print(f"done in {time.monotonic() - t0:.1f}s.", flush=True)
        except Exception as e:
            print(
                f"warm-up failed ({type(e).__name__}: {e}) — proceeding anyway.",
                flush=True,
            )

        results: list[dict] = []
        for i, case in enumerate(cases, 1):
            prompt, expected = case["prompt"], case["expected_tool"]
            try:
                picked, arguments, latency = await run_case(
                    client, args.ollama_url, args.model, prompt, ollama_tools
                )
            except Exception as e:
                picked, arguments, latency = f"<error: {type(e).__name__}>", None, 0.0
            mark = "✓" if picked == expected else "×"
            args_str = _format_arguments(arguments)
            tail = (picked or "<no tool_call>") + args_str
            # flush=True so GitHub Actions streams the line immediately.
            # Without it, Python's block-buffered stdout (pipe → tee)
            # holds every case's line until the whole suite finishes,
            # and the step looks like a black screen for 20+ minutes.
            print(
                f"  [{i:>3}/{len(cases)}] {mark} {latency:>5.1f}s  {prompt[:55]:<57} → {tail}",
                flush=True,
            )
            results.append({
                "prompt": prompt,
                "expected": expected,
                "picked": picked,
                "arguments": arguments,
                "latency": latency,
            })

    return summarise(results, threshold)


# Per-model inference speed on the 4 vCPU GHA runner differs by ~2×:
# qwen2.5:7b runs at ~3-4 tok/s, qwen2.5:14b at ~1.5-2 tok/s. A single
# tool-call round-trip is therefore 60-150 s on 14b — well over the
# old 120 s ceiling, which caused 11/13 failures on the 2026-04-20
# nightly to be `<error: ReadTimeout>` rather than real regressions.
# Pick a generous default for the larger model; callers can override
# with `--timeout` / `EVAL_TIMEOUT`.
# Bumped 2026-04-21 after catalog growth (22 tools → ~3000 desc +
# ~1400 schema tokens) pushed 2-vCPU GHA prefill past the old 120/300 s
# ceilings. Observed as every scored case failing `<error: ReadTimeout>`
# and warm-up itself dying on 7b. Current headroom covers worst-case
# CPU prefill; GPU cuts it back to <5 s so the numbers are harmless
# there.
_DEFAULT_TIMEOUTS_BY_MODEL_SIZE = {
    "14b": 480.0,   # was 300
    "7b":  240.0,   # was 120
}


def _default_timeout_for_model(model: str) -> float:
    """Pick a per-request timeout that fits the model's CPU inference speed.

    Matches on the parameter-count suffix in Ollama model names
    (`qwen2.5:14b`, `qwen2.5:7b`, `qwen2.5-coder:7b`, ...). Unknown
    sizes fall back to the 7b ceiling — safe for models this size or
    smaller, and at least produces a real error instead of silent
    `<ReadTimeout>` for larger unexpected models.
    """
    lowered = model.lower()
    for tag, seconds in _DEFAULT_TIMEOUTS_BY_MODEL_SIZE.items():
        if tag in lowered:
            return seconds
    return _DEFAULT_TIMEOUTS_BY_MODEL_SIZE["7b"]


def _slice_cases(cases: list[dict], index: int, count: int) -> list[dict]:
    """Stride-based sharding: case[0,count,2*count,...] → chunk 0.

    Stride (not contiguous) is deliberate — it spreads every intent
    family across every chunk, so the warm-up cost amortises evenly
    and per-chunk `pass_threshold` checks don't collapse when a
    chunk happens to inherit only radio-station cases.
    """
    if count <= 1:
        return cases
    if not (0 <= index < count):
        raise ValueError(f"chunk index {index} out of range [0, {count})")
    return cases[index::count]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--cases", default=str(Path(__file__).parent / "cases.yaml"))
    p.add_argument("--mcp-url", default=os.environ.get("MCP_URL", "http://127.0.0.1:8000/mcp"))
    p.add_argument("--ollama-url", default=os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434"))
    p.add_argument("--model", default=os.environ.get("OLLAMA_MODEL", "qwen2.5:7b"))
    # Timeout can be overridden via `--timeout` or `EVAL_TIMEOUT`.
    # Leaving it unset auto-picks a ceiling that matches the model size
    # — 300 s for 14b, 120 s for 7b — so the nightly matrix stops
    # false-failing on ReadTimeout for the slower row.
    timeout_env = os.environ.get("EVAL_TIMEOUT")
    p.add_argument(
        "--timeout",
        type=float,
        default=float(timeout_env) if timeout_env else None,
        help="Per-request timeout in seconds (default: 300 for 14b, 120 for 7b)",
    )
    # Sharding lets the matrix fan each model × schema combo into N
    # parallel chunks so the 14b-on-CPU row no longer bumps the
    # workflow's 45-min ceiling. Warm-up and model-pull still happen
    # per chunk — trade-off accepted, wall-clock wins.
    p.add_argument(
        "--chunk-index",
        type=int,
        default=int(os.environ.get("EVAL_CHUNK_INDEX", "0")),
        help="Zero-based chunk index (stride = chunk_count).",
    )
    p.add_argument(
        "--chunk-count",
        type=int,
        default=int(os.environ.get("EVAL_CHUNK_COUNT", "1")),
        help="Total number of chunks the case list is sharded into.",
    )
    args = p.parse_args()
    if args.timeout is None:
        args.timeout = _default_timeout_for_model(args.model)
    print(f"Per-request timeout: {args.timeout:.0f}s (model={args.model})")
    if args.chunk_count > 1:
        print(f"Running chunk {args.chunk_index + 1}/{args.chunk_count}")
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
