# ----------------------------
# Base image
# ----------------------------
# Use a lightweight Python 3.11 image
FROM python:3.11-slim

# ----------------------------
# Working directory
# ----------------------------
WORKDIR /app

# ----------------------------
# System dependencies
# ----------------------------
# Installs basic build tools and curl.
# The cleanup keeps the image small.
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential curl && \
    rm -rf /var/lib/apt/lists/*

# ----------------------------
# Python dependencies
# ----------------------------
# Copy dependency definition files first (for Docker layer caching)
COPY requirements.txt .

# Install dependencies 
RUN pip install --no-cache-dir --require-hashes -r requirements.txt


# ----------------------------
# Project files
# ----------------------------
# Copy the rest of the project code
RUN pip list | grep -E "lang|google|openai"
COPY . .

# ----------------------------
# Runtime environment
# ----------------------------
# Cloud Run sets the $PORT dynamically
ENV PORT=8080

# Optional but good practice: explicitly expose the port
EXPOSE 8080

# ----------------------------
# Command to start the application
# ----------------------------
# Run Uvicorn server for FastAPI
CMD ["uvicorn", "agent.app.main:app", "--host", "0.0.0.0", "--port", "8080"]
