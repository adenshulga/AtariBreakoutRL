FROM python:3.13-slim

# THIS IS DEVELOPMENT DOCKERFILE

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    fish \
    git \
    curl \
    vim \
    bash \
    ffmpeg \
    tmux \
    unzip \
    build-essential \
    python3-dev && \
    rm -rf /var/lib/apt/lists/*

ARG USER_ID=1000
ARG GROUP_ID=1000

# Create a group and user with the specified UID and GID
RUN addgroup --gid $GROUP_ID appgroup && \
    adduser --uid $USER_ID --gid $GROUP_ID --shell /bin/bash --disabled-password --gecos "" appuser

# Install sudo and grant privileges
RUN apt-get update && apt-get install -y sudo && \
    echo "appuser ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Create /app directory with proper ownership
RUN mkdir -p /app && chown -R appuser:appgroup /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Switch to the new user
USER appuser

# SHELL ["/usr/bin/fish", "-c"]

WORKDIR /app

# COPY --chown=appuser:appgroup . /app

# RUN uv venv .venv

# RUN uv sync

# EXPOSE 7860

# CMD scripts/launch_app.sh