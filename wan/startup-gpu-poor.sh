#!/usr/bin/env bash
set -euo pipefail

echo "🔧 Initializing Wan2GP container (GPU-constrained setup)..."

# Read env variable for auto-update
AUTO_UPDATE="${AUTO_UPDATE:-0}"

# Define cache and runtime paths
CACHE_DIR="/app/cache"
export HF_HOME="${CACHE_DIR}/huggingface"
export TORCH_HOME="${CACHE_DIR}/torch"
CKPTS_DIR="${CACHE_DIR}/ckpts"
LORA_DIR="${CACHE_DIR}/loras"
LORA_I2V_DIR="${CACHE_DIR}/loras_i2v"
OUTPUT_DIR="/app/output"

# Prepare directories
echo "📂 Creating runtime directories..."
mkdir -p "${HF_HOME}" "${TORCH_HOME}" "${CKPTS_DIR}" "${LORA_DIR}" "${LORA_I2V_DIR}" "${OUTPUT_DIR}"

# Extract application repo if needed
REPO_DIR="${CACHE_DIR}/repo"
if [ ! -d "$REPO_DIR" ]; then
    echo "📥 Extracting Wan2GP source..."
    mkdir -p "$REPO_DIR"
    tar -xzf Wan2GP.tar.gz --strip-components=1 -C "$REPO_DIR"
fi

# Optionally refresh repo if auto-update is enabled
if [[ "$AUTO_UPDATE" == "1" ]]; then
    echo "🔄 Updating Wan2GP repo..."
    git -C "$REPO_DIR" reset --hard
    git -C "$REPO_DIR" pull
fi

# Establish required symlinks
ln -sfn "${CKPTS_DIR}" "${REPO_DIR}/ckpts"
ln -sfn "${LORA_DIR}" "${REPO_DIR}/lora"
ln -sfn "${LORA_I2V_DIR}" "${REPO_DIR}/lora_i2v"
ln -sfn "${OUTPUT_DIR}" "${REPO_DIR}/gradio_outputs"

# Config file linkage (safe and bulletproof)
CONFIG_PATH="/app/config.json"
if [ ! -s "$CONFIG_PATH" ]; then
    echo "🛠️ Creating default config.json with safe defaults..."
    cat <<EOF > "$CONFIG_PATH"
{
    "version": "1.0",
    "attention_mode": "sdpa",
    "settings": {}
}
EOF
fi

ln -sfn "$CONFIG_PATH" "${REPO_DIR}/gradio_config.json"

# Setup Python virtual environment
VENV_DIR="${CACHE_DIR}/venv"
echo "🐍 Configuring Python venv..."
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR" --system-site-packages
fi
source "${VENV_DIR}/bin/activate"

# Upgrade base tools
pip install --no-cache-dir --upgrade pip wheel

# Install core and model dependencies
echo "📦 Installing project dependencies..."
pip install --no-cache-dir \
    packaging \
    torch==2.6.0 \
    torchvision \
    torchaudio \
    --index-url https://download.pytorch.org/whl/test/cu124

pip install --no-cache-dir -r "${REPO_DIR}/requirements.txt"
pip install --no-cache-dir \
    flash-attn==2.7.2.post1 \
    sageattention==1.0.6

# Build application launch args
WAN2GP_ARGS="--server-name 0.0.0.0 --server-port 7860 --compile --profile 1 --multiple-images --verbose 2"

# Launch the application
echo "🚀 Launching Wan2GP..."
cd "${REPO_DIR}"
exec python3 -u wgp.py ${WAN2GP_ARGS} 2>&1 | tee "${CACHE_DIR}/output.log"

echo "❌ Wan2GP application terminated."
