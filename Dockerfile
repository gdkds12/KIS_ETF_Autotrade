# -----------------------------------------------------------------------------
# Dockerfile for KIS ETF Autotrade Project
# -----------------------------------------------------------------------------

# --- Base Stage ---
# Use an official Python runtime as a parent image
# Choose a specific version for reproducibility (e.g., 3.11)
# Slim variant reduces image size
FROM python:3.11-slim AS base

# Set environment variables to prevent Python from writing pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory in the container
WORKDIR /app

# --- Builder Stage (for dependencies) ---
# Install build tools and Poetry
FROM base AS builder

# Install system dependencies that might be required by some Python packages
# (e.g., build-essential for C extensions, libpq-dev for psycopg2)
# Adjust based on your actual project dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential libpq-dev curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Poetry (using the recommended installer)
# Pinning the version is recommended for stable builds
ENV POETRY_VERSION=1.8.2
RUN curl -sSL https://install.python-poetry.org | python3 - --version ${POETRY_VERSION}
# Add poetry to PATH
ENV PATH="${PATH}:/root/.local/bin"

# Configure Poetry to not create virtual environments within the container,
# as we want dependencies installed directly into the path for the final image.
RUN poetry config virtualenvs.create false

# Copy only the dependency definition files first to leverage Docker cache
COPY poetry.lock* pyproject.toml ./

# Install project dependencies (excluding dev dependencies)
# --no-root: Skip installing the project package itself
# --no-interaction: Do not ask interactive questions
# --no-ansi: Disable ANSI output
RUN poetry install --no-root --no-interaction --no-ansi --without dev

# --- Final Stage ---
# Use the slim base image again for the final, smaller image
FROM base AS final

# Copy installed dependencies from the builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /root/.local/bin /root/.local/bin

# Copy the application source code and necessary config files
# Assumes the build context is the KID_ETF_Autotrade directory
COPY src ./src
COPY alembic.ini ./
COPY alembic ./alembic
# Add any other necessary files or directories here (e.g., static assets, templates)

# Expose the port the FastAPI app runs on (for documentation, actual mapping is in docker-compose)
EXPOSE 8000

# The command to run the application will be specified in docker-compose.yml
# depending on whether it's the FastAPI app or the Discord bot service.
# Example placeholder (not executed by default when using docker-compose command):
# CMD ["poetry", "run", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
