FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

COPY pyproject.toml uv.lock README.md /app/
COPY src /app/src
RUN uv sync --frozen --no-dev --no-install-project \
    && printf '%s\n' '#!/bin/sh' 'exec python -m cellmeasurement.cli "$@"' > /usr/local/bin/cellmeasurement \
    && chmod +x /usr/local/bin/cellmeasurement

RUN useradd --create-home --uid 10001 appuser && chown -R appuser:appuser /app
USER appuser

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app/src"
