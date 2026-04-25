FROM python:3.11-slim

# Prevent prompts during apt installation and ensure unbuffered python output
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

WORKDIR /app

# Install system dependencies
# git is required for whisperx to pull from github, ffmpeg for audio extraction
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg git && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Install PyTorch with specific CUDA 12.1 support first
# We do this before copying the rest of the application code to cache this heavy layer
RUN pip install --no-cache-dir torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

# Install GPU transcription and speaker diarization dependencies
RUN pip install --no-cache-dir whisperx pyannote.audio

# Install runtime dependencies from pyproject.toml without the package itself.
# This layer is cached as long as dependency versions don't change.
COPY pyproject.toml README.md ./
RUN pip install --no-cache-dir pydantic pydantic-settings structlog google-genai rich

# Now copy the application source and install the package
COPY trebek/ ./trebek/
RUN pip install --no-cache-dir .

# We run as root by default to avoid permission issues when mounting host volumes
# like input_videos and trebek.db. Users mapping to network drives should ensure
# POSIX locking support for SQLite WAL mode.
ENTRYPOINT ["trebek"]
