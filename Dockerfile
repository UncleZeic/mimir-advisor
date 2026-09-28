FROM python:3.11-slim

WORKDIR /app

# Dependencies first, to leverage Docker layer caching.
# sqlite-vec ships prebuilt wheels, so no compiler is needed.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Mimir doesn't run as root; safety first
RUN useradd -m mimir
USER mimir

CMD ["python", "src/main.py"]
