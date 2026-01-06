#!/bin/bash
set -e

echo "🔧 Starting OmniControlGP..."
echo "=================================================="

# Function to check GPU
check_gpu() {
    if command -v nvidia-smi &> /dev/null; then
        echo "🟢 GPU detected:"
        nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
        VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1)
        echo "📊 Total VRAM: ${VRAM} MB"
        
        if [ "$VRAM" -lt 12000 ]; then
            echo "⚠️  Warning: Less than 12GB VRAM detected. Consider using dockerfile-gpu-poor for better performance."
        fi
    else
        echo "❌ No GPU detected. This application requires NVIDIA GPU."
        exit 1
    fi
}

# Function to check models
check_models() {
    echo ""
    echo "🔍 Checking models..."
    
    MODEL_DIR="${MODEL_CACHE_DIR:-/app/models}"
    mkdir -p "$MODEL_DIR"
    
    # Check for FLUX.1-schnell base model
    if [ ! -d "$MODEL_DIR/models--black-forest-labs--FLUX.1-schnell" ]; then
        echo "📥 FLUX.1-schnell base model not found. It will be downloaded on first run (~34GB)."
        echo "    This may take some time depending on your internet connection..."
    else
        echo "✅ FLUX.1-schnell base model found"
    fi
    
    # Check for OminiControl LoRA weights
    if [ ! -d "$MODEL_DIR/models--Yuanshi--OminiControl" ]; then
        echo "📥 OminiControl LoRA weights not found. They will be downloaded on first run (~200MB)."
    else
        echo "✅ OminiControl LoRA weights found"
    fi
}

# Function to check available memory
check_memory() {
    echo ""
    echo "💾 System Memory:"
    free -h | grep -E "Mem:|Swap:"
}

# Function to set profile based on VRAM
set_profile() {
    if [ -z "$OMNI_PROFILE" ]; then
        # Use Profile 3 - optimized for exactly 12GB VRAM
        # Loads to RAM first (50GB), then VRAM (10-12GB)
        if [ "$VRAM" -ge 11000 ]; then
            export OMNI_PROFILE=3
            echo "🚀 Using Profile 3 (12GB VRAM - full RAM pinning + fast VRAM)"
        elif [ "$VRAM" -ge 7000 ]; then
            export OMNI_PROFILE=4
            echo "🚀 Using Profile 4 (8GB mode)"
        else
            export OMNI_PROFILE=5
            echo "🚀 Using Profile 5 (6GB mode)"
        fi
    else
        echo "🚀 Using Profile ${OMNI_PROFILE} (Custom)"
    fi
    
    echo "    Profile ${OMNI_PROFILE} selected - should generate images in 6-10 seconds"
    echo "    You can override by setting OMNI_PROFILE environment variable (1, 3, 4, or 5)"
}

# Main execution
check_gpu
check_models
check_memory
set_profile

echo ""
echo "=================================================="
echo "🌐 Starting Gradio interface on http://0.0.0.0:7860"
echo "=================================================="
echo ""

# Start the Gradio application
cd /app
exec python3 gradio_interface.py \
    --profile "${OMNI_PROFILE}" \
    --verbose "${OMNI_VERBOSE:-1}"
