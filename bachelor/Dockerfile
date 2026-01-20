FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl build-essential git \
    libglib2.0-0 libsm6 libxrender1 libxext6 libgl1 libegl1 libosmesa6-dev && \
    rm -rf /var/lib/apt/lists/*

RUN curl -sSL https://install.python-poetry.org | python3 -

# Add Poetry to PATH
ENV PATH="/root/.local/bin:$PATH"
ENV PYTHONPATH=/app

WORKDIR /app/myapp

COPY myapp/pyproject.toml myapp/poetry.lock* ./

# Install dependencies
RUN poetry install --no-root --only main

# Copy the full project
COPY . .

EXPOSE 7860


ENV IS_PROD=1
ENV SUPABASE_URL=""
ENV SUPABASE_KEY=""
ENV B2_KEY_ID=""
ENV B2_APP_KEY=""
ENV B2_BUCKET_NAME=""
ENV PINECONE_API_KEY=""
ENV PINECONE_REGION=""
ENV ENVIRONMENT="production"


CMD ["poetry", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860", "--reload"]