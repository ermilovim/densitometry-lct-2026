FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project \
    && .venv/bin/python -c "import torch; assert torch.version.cuda == '12.8', torch.version.cuda"

COPY . .
RUN uv sync --frozen

EXPOSE 8765

CMD ["uv", "run", "python", "viewer/server.py", "--host", "0.0.0.0", "--port", "8765"]
