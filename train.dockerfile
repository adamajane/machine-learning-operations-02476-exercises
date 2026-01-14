# Base image
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

WORKDIR /

COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml

ENV UV_LINK_MODE=copy
RUN --mount=type=cache,target=/root/.cache/uv uv sync --locked --no-install-project

COPY src/ src/
COPY data/ data/
COPY README.md README.md
COPY LICENSE LICENSE

ENTRYPOINT ["uv", "run", "src/mlops_mnist_classifier/train.py"]
