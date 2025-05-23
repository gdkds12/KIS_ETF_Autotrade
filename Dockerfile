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
    RUN apt-get update && \
        apt-get install -y --no-install-recommends build-essential libpq-dev curl git && \
        apt-get clean && rm -rf /var/lib/apt/lists/*  # git 추가
    
    # Copy only the requirements file first to leverage Docker cache
    COPY requirements.lock ./
    
    # Upgrade pip to the latest version (권장)
    RUN pip install --upgrade pip
    # Install project dependencies using pip
    # requirements.lock에 tavily가 포함되어 있으므로 별도 명령 없이 자동 설치됩니다.
    # Use --no-cache-dir to reduce image size
    # Pin httpx to a version compatible with OpenAI SDK
    RUN pip install --pre --no-cache-dir -r requirements.lock && \
        pip install --no-cache-dir "httpx<0.24.0"
    
    # --- Final Stage ---
    FROM base AS final
    
    # Copy installed dependencies from the builder stage
    COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
    
    # Copy the application source code
    COPY src ./src
    
    # Expose the application port
    EXPOSE 8000
    
    # The startup command should be defined in docker-compose.yml
    