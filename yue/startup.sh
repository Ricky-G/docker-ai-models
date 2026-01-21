#!/bin/bash
set -e

echo "🔧 Starting YuE GPU-Poor (Profile ${YUE_PROFILE:-3})..."

# Setup persistent storage
export HF_HOME="${HF_HOME:-/app/models}"
export TORCH_HOME="${TORCH_HOME:-/app/models}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/app/models/pip}"
mkdir -p "$HF_HOME" "$TORCH_HOME" "$PIP_CACHE_DIR" /app/output

# Check GPU
if command -v nvidia-smi &> /dev/null; then
    echo "🟢 GPU:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "❌ No GPU detected"
    exit 1
fi

# Install dependencies (uses cached wheels from D:\_Models\yue for speed)
echo "📦 Installing dependencies (uses cached wheels if available)..."
pip3 install --cache-dir="$PIP_CACHE_DIR" torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip3 install --cache-dir="$PIP_CACHE_DIR" -r /app/requirements.txt
pip3 install --cache-dir="$PIP_CACHE_DIR" flash-attn --no-build-isolation 2>/dev/null || echo "⚠️  Flash attention unavailable, using SDPA fallback"
echo "✅ Dependencies ready"

echo "🎵 Starting Gradio interface on http://0.0.0.0:7860"
echo "   Profile 3: 12GB VRAM + 8-bit quantization"
echo "   Models will download from HuggingFace on first use"
echo "   Models cached to: $HF_HOME"

exec python3 -u /app/gradio_interface.py
