FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-server.txt ./
RUN pip install --no-cache-dir -r requirements-server.txt

COPY environment ./environment
COPY demo ./demo
COPY data ./data
COPY eval ./eval
COPY logs ./logs
COPY training ./training
COPY openenv.yaml ./openenv.yaml
COPY README.md ./README.md

RUN useradd --create-home --shell /usr/sbin/nologin appuser \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:7860/ || exit 1

CMD ["python", "demo/app.py"]
