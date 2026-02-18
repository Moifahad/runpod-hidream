FROM nvidia/cuda:12.8.0-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# ---------------------------------------------------------------------------
# System dependencies
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-dev python3-venv \
    git wget build-essential ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Use a venv to avoid pip system-packages issues
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --upgrade pip setuptools wheel

# ---------------------------------------------------------------------------
# Install torch first (large download, cached as its own layer)
# ---------------------------------------------------------------------------
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cu128 \
    torch

# ---------------------------------------------------------------------------
# Install flash-attn (required by HiDream)
# ---------------------------------------------------------------------------
RUN pip install --no-cache-dir -U flash-attn --no-build-isolation

# ---------------------------------------------------------------------------
# Install diffusers from source (HiDreamImagePipeline needs latest)
# ---------------------------------------------------------------------------
RUN git clone --depth=1 https://github.com/huggingface/diffusers.git /tmp/diffusers \
    && pip install --no-cache-dir /tmp/diffusers \
    && rm -rf /tmp/diffusers

# ---------------------------------------------------------------------------
# Install remaining dependencies
# ---------------------------------------------------------------------------
RUN pip install --no-cache-dir \
    runpod>=1.7.0 \
    transformers>=4.47.0 \
    accelerate>=0.34.0 \
    sentencepiece \
    protobuf

# ---------------------------------------------------------------------------
# Copy handler
# ---------------------------------------------------------------------------
COPY handler.py /app/handler.py

CMD ["python", "/app/handler.py"]
