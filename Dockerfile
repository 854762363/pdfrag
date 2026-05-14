# PDFRAG Docker image
# Build: docker build -t pdfrag .
# Run:   docker-compose up

FROM python:3.12-slim

WORKDIR /app

# System deps for pdf2image (poppler) + PaddlePaddle
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code
COPY src/ ./src/
COPY core/ ./core/
COPY tests/ ./tests/
COPY data/ ./data/
COPY README.md .

# Create data dirs
RUN mkdir -p data/uploads data/chroma data/cache

EXPOSE 8000

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
