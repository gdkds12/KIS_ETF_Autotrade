# -----------------------------------------------------------------------------
# Dockerfile for KIS ETF Autotrade Project (Using requirements.lock)
# -----------------------------------------------------------------------------

# --- Base Stage ---
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /app

# --- Builder Stage (for dependencies) ---
# Install build tools needed for some packages
FROM base AS builder

# Install system dependencies that might be required by some Python packages
# (e.g., build-essential for C extensions, libpq-dev for psycopg2)
# Keep curl if needed for other steps, otherwise can remove
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential libpq-dev curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy only the requirements file first to leverage Docker cache
# Assuming requirements.lock contains all direct and transitive dependencies pinned
COPY requirements.lock ./

# Install project dependencies using pip
# Use --no-cache-dir to reduce image size
# Use --require-hashes if your requirements.lock includes hashes for security
RUN pip install --no-cache-dir -r requirements.lock
# If requirements.lock doesn't pin everything, you might need requirements.txt first:
# COPY requirements.txt ./
# RUN pip install --no-cache-dir -r requirements.txt

# --- Final Stage ---
FROM base AS final

# Copy installed dependencies from the builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
# No need to copy poetry binary path anymore

# Copy the application source code
COPY src ./src
# COPY alembic.ini ./  <-- Removed as alembic.ini is not available
COPY alembic ./alembic

EXPOSE 8000

# CMD will be specified in docker-compose.yml
