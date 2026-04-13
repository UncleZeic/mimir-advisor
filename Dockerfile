# Use a slim Python base to keep the robot lean
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies for audio and networking
RUN apt-get update && apt-get install -y \
    build-essential \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy only the requirements first (to leverage Docker caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the logic
COPY . .

# Mimir doesn't run as root; safety first
RUN useradd -m mimir
USER mimir

# The entry point for the Advisor
CMD ["python", "src/main.py"]