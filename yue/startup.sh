#!/usr/bin/env bash
set -euo pipefail

echo "🔧 Initializing YuE container (Official Implementation)..."

# -- Configuration from environment (with official YuE defaults) --
YUE_CUDA_IDX="${YUE_CUDA_IDX:-0}"
YUE_AUTO_UPDATE="${YUE_AUTO_UPDATE:-0}"
YUE_STAGE1_MODEL="${YUE_STAGE1_MODEL:-m-a-p/YuE-s1-7B-anneal-en-cot}"
YUE_STAGE2_MODEL="${YUE_STAGE2_MODEL:-m-a-p/YuE-s2-1B-general}"
YUE_MAX_NEW_TOKENS="${YUE_MAX_NEW_TOKENS:-3000}"
YUE_REPETITION_PENALTY="${YUE_REPETITION_PENALTY:-1.1}"
YUE_STAGE2_BATCH_SIZE="${YUE_STAGE2_BATCH_SIZE:-4}"
YUE_RUN_N_SEGMENTS="${YUE_RUN_N_SEGMENTS:-2}"
YUE_USE_AUDIO_PROMPT="${YUE_USE_AUDIO_PROMPT:-0}"
YUE_USE_DUAL_TRACKS_PROMPT="${YUE_USE_DUAL_TRACKS_PROMPT:-0}"
YUE_AUDIO_PROMPT_PATH="${YUE_AUDIO_PROMPT_PATH:-}"
YUE_VOCAL_TRACK_PROMPT_PATH="${YUE_VOCAL_TRACK_PROMPT_PATH:-}"
YUE_INSTRUMENTAL_TRACK_PROMPT_PATH="${YUE_INSTRUMENTAL_TRACK_PROMPT_PATH:-}"
YUE_PROMPT_START_TIME="${YUE_PROMPT_START_TIME:-0.0}"
YUE_PROMPT_END_TIME="${YUE_PROMPT_END_TIME:-30.0}"
YUE_KEEP_INTERMEDIATE="${YUE_KEEP_INTERMEDIATE:-1}"
YUE_DISABLE_OFFLOAD_MODEL="${YUE_DISABLE_OFFLOAD_MODEL:-0}"
YUE_SEED="${YUE_SEED:-42}"
YUE_MODE="${YUE_MODE:-cli}"

# -- Cache and runtime setup --
export HF_HOME="/app/cache/huggingface"
export TORCH_HOME="/app/cache/torch"
mkdir -p "$HF_HOME" "$TORCH_HOME" /app/output

echo "📂 Cache directories configured:"
echo "  HuggingFace: $HF_HOME"
echo "  PyTorch: $TORCH_HOME"
echo "  Output: /app/output"

# -- Activate conda environment --
echo "🐍 Activating conda environment..."
source /opt/conda/etc/profile.d/conda.sh
conda activate yue

# -- Auto-update repository if enabled --
if [ "$YUE_AUTO_UPDATE" == "1" ]; then
    echo "🔄 Updating YuE repository..."
    cd /app/YuE
    git reset --hard
    git pull
    git lfs pull
    
    # Re-install requirements if updated
    pip install --no-cache-dir -r requirements.txt
fi

# -- Verify xcodec_mini_infer is available --
XCODEC_PATH="/app/YuE/inference/xcodec_mini_infer"
if [ ! -d "$XCODEC_PATH" ]; then
    echo "📥 Cloning xcodec_mini_infer..."
    git clone https://huggingface.co/m-a-p/xcodec_mini_infer "$XCODEC_PATH"
elif [ "$YUE_AUTO_UPDATE" == "1" ]; then
    echo "🔄 Updating xcodec_mini_infer..."
    cd "$XCODEC_PATH"
    git reset --hard
    git pull
fi

# -- Ensure prompt files exist with defaults --
GENRE_FILE="/app/YuE/prompt_egs/genre.txt"
LYRICS_FILE="/app/YuE/prompt_egs/lyrics.txt"

if [ ! -f "$GENRE_FILE" ]; then
    echo "📝 Creating default genre.txt..."
    mkdir -p "$(dirname "$GENRE_FILE")"
    echo "inspiring female uplifting pop airy vocal electronic bright vocal" > "$GENRE_FILE"
fi

if [ ! -f "$LYRICS_FILE" ]; then
    echo "📝 Creating default lyrics.txt..."
    mkdir -p "$(dirname "$LYRICS_FILE")"
    cat > "$LYRICS_FILE" << 'EOF'
[verse]
Walking through the city lights tonight
Everything seems so bright
Dreams are calling out my name
Nothing will ever be the same

[chorus]
We're rising up, we're breaking free
This is who we're meant to be
With every step we take today
We're lighting up our own way
EOF
fi

# -- Build official infer.py arguments --
ARGS="--cuda_idx ${YUE_CUDA_IDX} \
      --stage1_model ${YUE_STAGE1_MODEL} \
      --stage2_model ${YUE_STAGE2_MODEL} \
      --genre_txt ${GENRE_FILE} \
      --lyrics_txt ${LYRICS_FILE} \
      --run_n_segments ${YUE_RUN_N_SEGMENTS} \
      --stage2_batch_size ${YUE_STAGE2_BATCH_SIZE} \
      --output_dir /app/output \
      --max_new_tokens ${YUE_MAX_NEW_TOKENS} \
      --repetition_penalty ${YUE_REPETITION_PENALTY} \
      --seed ${YUE_SEED}"

# -- Add optional features --
if [ "$YUE_KEEP_INTERMEDIATE" == "1" ]; then
    ARGS="${ARGS} --keep_intermediate"
fi

if [ "$YUE_DISABLE_OFFLOAD_MODEL" == "1" ]; then
    ARGS="${ARGS} --disable_offload_model"
fi

# -- Add ICL support based on configuration --
if [ "$YUE_USE_DUAL_TRACKS_PROMPT" == "1" ]; then
    if [ -n "$YUE_VOCAL_TRACK_PROMPT_PATH" ] && [ -n "$YUE_INSTRUMENTAL_TRACK_PROMPT_PATH" ]; then
        echo "🎵 Dual-track ICL mode enabled"
        ARGS="${ARGS} --use_dual_tracks_prompt \
              --vocal_track_prompt_path ${YUE_VOCAL_TRACK_PROMPT_PATH} \
              --instrumental_track_prompt_path ${YUE_INSTRUMENTAL_TRACK_PROMPT_PATH} \
              --prompt_start_time ${YUE_PROMPT_START_TIME} \
              --prompt_end_time ${YUE_PROMPT_END_TIME}"
    else
        echo "⚠️  Warning: Dual-track ICL enabled but paths not provided"
    fi
elif [ "$YUE_USE_AUDIO_PROMPT" == "1" ]; then
    if [ -n "$YUE_AUDIO_PROMPT_PATH" ]; then
        echo "🎵 Single-track ICL mode enabled"
        ARGS="${ARGS} --use_audio_prompt \
              --audio_prompt_path ${YUE_AUDIO_PROMPT_PATH} \
              --prompt_start_time ${YUE_PROMPT_START_TIME} \
              --prompt_end_time ${YUE_PROMPT_END_TIME}"
    else
        echo "⚠️  Warning: Audio prompt enabled but path not provided"
    fi
fi

# -- Display configuration --
echo "⚙️ YuE Configuration (Official infer.py):"
echo "  Mode: $YUE_MODE"
echo "  CUDA Device: $YUE_CUDA_IDX"
echo "  Stage 1 Model: $YUE_STAGE1_MODEL"
echo "  Stage 2 Model: $YUE_STAGE2_MODEL"
echo "  Max Tokens: $YUE_MAX_NEW_TOKENS"
echo "  Batch Size: $YUE_STAGE2_BATCH_SIZE"
echo "  Segments: $YUE_RUN_N_SEGMENTS"
echo "  Dual-track ICL: $YUE_USE_DUAL_TRACKS_PROMPT"
echo "  Single-track ICL: $YUE_USE_AUDIO_PROMPT"
echo "  Auto-Update: $YUE_AUTO_UPDATE"
echo "  Seed: $YUE_SEED"

# -- Navigate to inference directory and launch --
echo "🚀 Launching YuE inference (Official implementation)..."
cd /app/YuE/inference || exit 1

# Check the mode and launch accordingly
if [ "$YUE_MODE" == "web" ]; then
    echo "🌐 Starting Gradio web interface..."
    echo "📝 Web interface will be available at: http://0.0.0.0:7860"
    echo "📁 Output will be saved to: /app/output"
    echo ""
    
    # Launch Gradio interface
    cd /app
    exec python -u gradio_interface.py --host 0.0.0.0 --port 7860 2>&1 | tee "/app/cache/yue-gradio.log"
else
    echo "💻 Starting CLI mode..."
    echo "📝 Command: python infer.py ${ARGS}"
    echo "📁 Output will be saved to: /app/output"
    echo ""
    
    exec python -u infer.py ${ARGS} 2>&1 | tee "/app/cache/yue-output.log"
fi
