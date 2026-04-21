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
| `get_today(timezone="UTC")`                        | Today's date, weekday and the timezone used as anchor. Call this first when the user asks about "today" / "tomorrow" / a weekday.                    |
| `geocode_city(city, country_code=None)`            | Resolve a city name or postal code to lat/lon and timezone. `country_code` (ISO-3166-1 alpha-2) disambiguates homonyms and zip-code collisions.      |
| `list_cities(query, country_code=None, limit=5)`   | Return every geocoding candidate for an ambiguous query (e.g. "Springfield"). Lets the caller pick one on the user's behalf or ask for clarification. |
| `get_current_weather(city, country_code=None)`     | Current temperature, humidity, wind, conditions.                                                                                                      |
| `get_forecast(city, days=7, country_code=None)`    | Daily forecast, 1–16 days ahead. Each entry carries `day_label` ("today", "tomorrow", "in N days") anchored to the city's local timezone.            |

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
hostname.

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

## Project files

- `server.py` — the MCP server (FastMCP). Transport is controlled by
  `MCP_TRANSPORT` (`stdio` | `streamable-http`), port by `MCP_PORT`.
- `pyproject.toml` / `uv.lock` — dependencies (`mcp[cli]`, `httpx`).
- `mcphost.config.json` — `mcphost` config for Option B.
- `Dockerfile` / `.dockerignore` — multi-stage image built with `uv`,
  runs as non-root with a read-only root filesystem.
- `k8s/` — Kustomize manifests: Deployment, Service, NetworkPolicy,
  Secret template, and an optional Traefik IngressRoute.

## License

MIT.
