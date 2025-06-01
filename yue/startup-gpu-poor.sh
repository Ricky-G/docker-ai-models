#!/usr/bin/env bash
set -euo pipefail

echo "🔧 Booting up YuEGP container (GPU-constrained setup)..."

# -- Config from environment (with defaults) --
YUEGP_PROFILE="${YUEGP_PROFILE:-1}"
YUEGP_CUDA_IDX="${YUEGP_CUDA_IDX:-0}"
YUEGP_ENABLE_ICL="${YUEGP_ENABLE_ICL:-0}"
YUEGP_TRANSFORMER_PATCH="${YUEGP_TRANSFORMER_PATCH:-0}"
YUEGP_AUTO_UPDATE="${YUEGP_AUTO_UPDATE:-0}"
YUEGP_SERVER_USER="${YUEGP_SERVER_USER:-}"
YUEGP_SERVER_PASSWORD="${YUEGP_SERVER_PASSWORD:-}"

# -- Caching setup --
CACHE_ROOT="/workspace/cache"
export HF_HOME="${CACHE_ROOT}/huggingface"
export TORCH_HOME="${CACHE_ROOT}/torch"
mkdir -p "${HF_HOME}" "${TORCH_HOME}" /workspace/output

# -- Unpack model code --
YUEGP_HOME="${CACHE_ROOT}/YuEGP"
XCODEC_HOME="${CACHE_ROOT}/xcodec_mini_infer"

if [ ! -d "$YUEGP_HOME" ]; then
    echo "📦 Extracting YuEGP..."
    mkdir -p "$YUEGP_HOME"
    tar -xzf YuEGP.tar.gz --strip-components=1 -C "$YUEGP_HOME"
fi

if [ "$YUEGP_AUTO_UPDATE" == "1" ]; then
    git -C "$YUEGP_HOME" reset --hard && git -C "$YUEGP_HOME" pull
fi

if [ ! -d "$XCODEC_HOME" ]; then
    echo "📦 Extracting xcodec_mini_infer..."
    mkdir -p "$XCODEC_HOME"
    tar -xzf xcodec_mini_infer.tar.gz --strip-components=1 -C "$XCODEC_HOME"
fi

if [ "$YUEGP_AUTO_UPDATE" == "1" ]; then
    git -C "$XCODEC_HOME" reset --hard && git -C "$XCODEC_HOME" pull
fi

# -- Link inference dependency --
ln -sfn "$XCODEC_HOME" "${YUEGP_HOME}/inference/xcodec_mini_infer"

# -- Virtual environment and dependency install --
VENV_HOME="${CACHE_ROOT}/venv"
if [ ! -d "$VENV_HOME" ]; then
    echo "🐍 Setting up Python venv..."
    python3 -m venv "$VENV_HOME" --system-site-packages
fi

source "$VENV_HOME/bin/activate"

echo "📦 Installing dependencies..."
pip install --no-cache-dir --upgrade pip wheel
pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/test/cu124
pip install --no-cache-dir --root-user-action=ignore -r "${YUEGP_HOME}/requirements.txt"
pip install --no-cache-dir --root-user-action=ignore flash-attn --no-build-isolation

# -- Optional Transformer patch --
if [ "$YUEGP_TRANSFORMER_PATCH" == "1" ]; then
    echo "🧪 Applying transformer patch..."
    ln -sfn "$VENV_HOME" "${YUEGP_HOME}/venv"
    (cd "$YUEGP_HOME" && source patchtransformers.sh)
fi

# -- Build run command --
ARGS="--profile ${YUEGP_PROFILE} \
      --cuda_idx ${YUEGP_CUDA_IDX} \
      --output_dir /workspace/output \
      --keep_intermediate \
      --server_name 0.0.0.0 \
      --server_port 7860"

if [ "$YUEGP_ENABLE_ICL" == "1" ]; then
    ARGS="${ARGS} --icl"
fi

# -- Launch service --
echo "🚀 Launching YuEGP..."
cd "${YUEGP_HOME}/inference" || exit 1
exec python3 -u gradio_server.py ${ARGS} 2>&1 | tee "${CACHE_ROOT}/output.log"
