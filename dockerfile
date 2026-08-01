FROM python:3.12-slim

WORKDIR /app

#COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-cache --no-dev

COPY . .

RUN mkdir -p data .store

RUN uv run --no-dev python -c "from langchain_qdrant import FastEmbedSparse; FastEmbedSparse(model_name='Qdrant/bm25')"

ENTRYPOINT ["uv", "run", "--no-dev"]
CMD ["main.py"]