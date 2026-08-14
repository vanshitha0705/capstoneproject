FROM python:3.11-slim

WORKDIR /app

# System deps needed by chromadb / sentence-transformers wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Build the ChromaDB index at image build time so the container is ready to
# serve immediately on `docker run` (MOCK_LLM defaults to "1", so this step
# needs no API key or network access to an LLM provider).
RUN python ingest.py

EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
