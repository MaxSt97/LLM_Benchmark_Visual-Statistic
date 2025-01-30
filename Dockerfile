# Basis-Image
FROM python:3.10-slim

# Arbeitsverzeichnis
WORKDIR /app

# Nur tk-dev installieren
RUN apt-get update && apt-get install -y --no-install-recommends \
    tk-dev \
    && rm -rf /var/lib/apt/lists/*

# Kopieren der requirements.txt
COPY requirements.txt /app/

# Installieren der Python-Abhängigkeiten
RUN pip install --no-cache-dir -r requirements.txt

# Kopieren des restlichen Codes
COPY . /app/

# Standardbefehl (anpassen, falls dein Hauptskript anders heißt)
CMD ["python", "Statistical_Tests.py"]