# syntax=docker/dockerfile:1.7

# --- Stage 1: build the virtualenv with uv ---
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Install deps first (better layer caching)
COPY pyproject.toml uv.lock* ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev || \
    uv sync --no-install-project --no-dev

# Source files — server.py plus the fat-router helpers it lazily
# imports when `MCP_ROUTER_MODE` is `fat_tools` or `fat_tools_lean`
# (the latter is now the default — see docs/tool-catalog-scaling.md).
# Glob picks up fat_tools.py, fat_tools_lean.py, fat_tools_map.py
# without having to touch the Dockerfile every time a new router
# variant lands.
COPY server.py fat_tools*.py ./

# --- Stage 2: runtime ---
FROM python:3.12-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH" \
    MCP_TRANSPORT=streamable-http \
    MCP_HOST=0.0.0.0 \
    MCP_PORT=8000

# Non-root user
RUN groupadd --system --gid 1000 app && \
    useradd  --system --uid 1000 --gid 1000 --home-dir /app --shell /usr/sbin/nologin app

WORKDIR /app
COPY --from=builder --chown=1000:1000 /app /app

USER 1000:1000
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request,sys; \
        r=urllib.request.urlopen('http://127.0.0.1:8000/mcp',timeout=3); \
        sys.exit(0 if r.status in (200,400,405,406) else 1)" || exit 1

ENTRYPOINT ["python", "server.py"]
