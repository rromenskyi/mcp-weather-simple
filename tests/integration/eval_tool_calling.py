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


def summarise(results: list[dict], threshold: float) -> int:
    """Print a pass/fail table grouped by intent family, return exit code."""
    total = len(results)
    hits = sum(1 for r in results if r["picked"] == r["expected"])
    rate = hits / total if total else 0.0

    # Family = common prefix of the expected tool (e.g. `get_weather_*`).
    by_family: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        family = "_".join(r["expected"].split("_")[:2])  # get_weather, get_current, list_radio, …
        by_family[family].append(r)

    print()
    print(f"{'family':<30} {'hits':>6} / {'total':>5}  {'rate':>6}")
    print("-" * 58)
    for family, rows in sorted(by_family.items()):
        fhits = sum(1 for r in rows if r["picked"] == r["expected"])
        ftotal = len(rows)
        print(f"{family:<30} {fhits:>6} / {ftotal:>5}  {fhits / ftotal:>6.1%}")
    print("-" * 58)
    print(f"{'OVERALL':<30} {hits:>6} / {total:>5}  {rate:>6.1%}   (threshold {threshold:.0%})")

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
    cases = cfg["cases"]
    threshold = cfg.get("pass_threshold", 0.70)

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

        # Warm-up: first real chat-with-tools call pulls the model into
        # Ollama's RAM, which easily takes 30–60 s on a CPU runner. If
        # we skip it, the first N scored cases all time out and drag
        # the hit-rate with them — which is a measurement artifact, not
        # a regression of the tool descriptions. Discard the result;
        # we only care about its side-effect (a warm model).
        print("Warming up model (first chat call loads it into RAM)...", end=" ", flush=True)
        try:
            _, _, warm_latency = await run_case(
                client, args.ollama_url, args.model,
                "Hello, this is a warm-up request. Please reply with 'ok'.",
                ollama_tools,
            )
            print(f"done in {warm_latency:.1f}s.")
        except Exception as e:
            # Warm-up failures are informational — the scored loop below
            # will surface real problems with its own timeout handling.
            print(f"warm-up failed ({type(e).__name__}: {e}) — proceeding anyway.")

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
            print(f"  [{i:>3}/{len(cases)}] {mark} {latency:>5.1f}s  {prompt[:55]:<57} → {tail}")
            results.append({
                "prompt": prompt,
                "expected": expected,
                "picked": picked,
                "arguments": arguments,
                "latency": latency,
            })

    return summarise(results, threshold)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--cases", default=str(Path(__file__).parent / "cases.yaml"))
    p.add_argument("--mcp-url", default=os.environ.get("MCP_URL", "http://127.0.0.1:8000/mcp"))
    p.add_argument("--ollama-url", default=os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434"))
    p.add_argument("--model", default=os.environ.get("OLLAMA_MODEL", "qwen2.5:7b"))
    p.add_argument("--timeout", type=float, default=120.0, help="Per-request timeout in seconds")
    args = p.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
