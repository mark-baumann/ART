# ═══════════════════════════════════════════════════════════════
# Dockerfile — Standard-Template für alle Streamlit-Apps
# ═══════════════════════════════════════════════════════════════
# Kopiere diese Datei in jedes App-Repo und passe PORT an.

FROM python:3.12-slim

WORKDIR /app

# System-Abhängigkeiten
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python-Abhängigkeiten
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App-Code
COPY . .

# Port (pro App anpassen: 8501-8519)
ARG PORT=8519
EXPOSE $PORT

# Healthcheck
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:${PORT}/_stcore/health')"

# Streamlit
CMD streamlit run app/app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true
