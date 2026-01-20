FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/root/.local/bin:$PATH" \
    PYTHONPATH=/app


RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl build-essential git \
    libglib2.0-0 libsm6 libxrender1 libxext6 libgl1 libegl1 libosmesa6-dev && \
    rm -rf /var/lib/apt/lists/*


COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app/myapp


COPY requirements.txt ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system --no-cache-dir -r requirements.txt

RUN uv pip install --system --upgrade PyOpenGL PyOpenGL_accelerate

COPY . .

EXPOSE 8000

ENV IS_PROD=0

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]