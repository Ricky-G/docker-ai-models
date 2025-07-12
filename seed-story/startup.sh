#!/bin/bash

# SEED-Story Docker Startup Script
# Supports both web interface and CLI modes

set -e

echo "🎬 Starting SEED-Story Multimodal Story Generation..."
echo "==================================================="

# Function to check GPU availability
check_gpu() {
    if nvidia-smi > /dev/null 2>&1; then
        echo "✅ GPU detected: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)"
        export CUDA_VISIBLE_DEVICES=0
    else
        echo "⚠️  No GPU detected - SEED-Story requires CUDA GPU"
        echo "   Please ensure you're running with --gpus all flag"
        exit 1
    fi
}

# Function to check model files
check_models() {
    echo "📁 Checking model directories..."
    
    # Use mounted volume for models if available, otherwise use container directory
    if [ -d "/app/models" ] && [ "$(ls -A /app/models 2>/dev/null)" ]; then
        MODELS_DIR="/app/models"
        echo "✅ Using mounted models directory: $MODELS_DIR"
    else
        MODELS_DIR="/app/pretrained"
        echo "⚠️  No mounted models directory found, using container directory: $MODELS_DIR"
    fi
    
    # Check for pretrained models
    if [ ! -d "$MODELS_DIR" ]; then
        echo "🔄 Creating models directory..."
        mkdir -p "$MODELS_DIR"
    fi
    
    # Export for use by model downloader
    export SEED_STORY_MODELS_DIR="$MODELS_DIR"
    
    # Check if essential models exist
    local required_models=(
        "stable-diffusion-xl-base-1.0"
        "Llama-2-7b-hf"
        "Qwen-VL-Chat"
    )
    
    local missing_models=()
    for model in "${required_models[@]}"; do
        if [ ! -d "$MODELS_DIR/$model" ]; then
            missing_models+=("$model")
        fi
    done
    
    if [ ${#missing_models[@]} -gt 0 ]; then
        echo "⚠️  Missing required models: ${missing_models[*]}"
        echo "   Use model_downloader.py to download missing models"
        echo "   Downloading to: $MODELS_DIR"
        echo "   Starting model downloader..."
        python3 /app/model_downloader.py
    else
        echo "✅ All required models found in $MODELS_DIR"
    fi
}

# Function to initialize SEED-Story environment
init_environment() {
    echo "🔧 Initializing SEED-Story environment..."
    
    # Set Python path
    export PYTHONPATH="/app:$PYTHONPATH"
    
    # Create output directories
    mkdir -p /app/data/output /app/data/input /app/data/temp
    
    # Use the same models directory as set in check_models
    MODELS_DIR=${SEED_STORY_MODELS_DIR:-"/app/pretrained"}
    
    # Skip VIT extraction - script not available in this setup
    echo "⚠️  Skipping Qwen VIT extraction (script not available)"
    
    echo "✅ Environment initialized"
}

# Function to start web interface
start_web_interface() {
    echo "🌐 Starting Gradio web interface on port 7860..."
    echo "   Access the interface at: http://localhost:7860"
    echo "   Use docker run -p 7860:7860 to map the port"
    
    cd /app
    
    # Use simple interface to avoid complex dependency issues
    echo "🔧 Starting with simple interface (avoiding dependency conflicts)..."
    # Try the simple comic generator first
    if [ -f "simple_comic_generator.py" ]; then
        echo "🎨 Using simple comic generator..."
        python3 simple_comic_generator.py
    else
        echo "📝 Falling back to minimal gradio..."
        python3 minimal_gradio.py
    fi
}

# Function to start CLI mode
start_cli_mode() {
    echo "💻 Starting CLI mode..."
    echo "   Available scripts:"
    echo "   - src/inference/gen_george.py (multimodal story generation)"
    echo "   - src/inference/vis_george_sink.py (story visualization)"
    
    # If specific script is provided as argument, run it
    if [ $# -gt 0 ]; then
        echo "🚀 Running: $*"
        cd /app
        python3 "$@"
    else
        # Interactive bash shell
        echo "🐚 Dropping to interactive shell..."
        cd /app
        exec /bin/bash
    fi
}

# Function to display help
show_help() {
    cat << EOF
SEED-Story Docker Container

Environment Variables:
  SEED_STORY_MODE     - Set to 'web' (default) or 'cli'
  CUDA_VISIBLE_DEVICES - GPU device to use (default: 0)

Usage Examples:
  # Web interface (default)
  docker run --gpus all -p 7860:7860 seed-story

  # CLI mode with specific script
  docker run --gpus all -e SEED_STORY_MODE=cli seed-story src/inference/gen_george.py

  # Interactive CLI
  docker run --gpus all -e SEED_STORY_MODE=cli -it seed-story

Available Scripts:
  - src/inference/gen_george.py        # Generate multimodal stories
  - src/inference/vis_george_sink.py   # Visualize story with attention
  - model_downloader.py                # Download required models

Model Requirements:
  - stable-diffusion-xl-base-1.0 (~7GB)
  - Llama-2-7b-hf (~13GB)
  - Qwen-VL-Chat (~10GB)
  - SEED-Story checkpoints (~2GB)

Total storage: ~32GB + working space
EOF
}

# Main execution
main() {
    # Handle help
    if [[ "$1" == "--help" || "$1" == "-h" ]]; then
        show_help
        exit 0
    fi
    
    # Check prerequisites
    check_gpu
    check_models
    init_environment
    
    # Determine mode
    MODE=${SEED_STORY_MODE:-web}
    
    case "$MODE" in
        "web")
            start_web_interface
            ;;
        "cli")
            shift # Remove mode from arguments if passed
            start_cli_mode "$@"
            ;;
        *)
            echo "❌ Invalid mode: $MODE"
            echo "   Set SEED_STORY_MODE to 'web' or 'cli'"
            exit 1
            ;;
    esac
}

# Handle script termination
trap 'echo "🛑 Shutting down SEED-Story..."; exit 0' SIGTERM SIGINT

# Execute main function with all arguments
main "$@"