#!/usr/bin/env python3
"""
YuE Gradio Web Interface
A simplified web interface for YuE music generation using Gradio.
Based on the official YuE implementation with support for both command-line and web modes.
"""

import gradio as gr
import os
import subprocess
import tempfile
import json
import shutil
import glob
from pathlib import Path
import threading
import queue
import time
import signal
import sys

# Global variables
generation_process = None
log_queue = queue.Queue()
process_lock = threading.Lock()

# Configuration
YUE_PATH = "/app/YuE"
OUTPUT_DIR = "/app/output"
CACHE_DIR = "/app/cache"

# Default values from environment variables
DEFAULT_STAGE1_MODEL = os.getenv("YUE_STAGE1_MODEL", "m-a-p/YuE-s1-7B-anneal-en-cot")
DEFAULT_STAGE2_MODEL = os.getenv("YUE_STAGE2_MODEL", "m-a-p/YuE-s2-1B-general")
DEFAULT_CUDA_IDX = int(os.getenv("YUE_CUDA_IDX", "0"))
DEFAULT_MAX_NEW_TOKENS = int(os.getenv("YUE_MAX_NEW_TOKENS", "3000"))
DEFAULT_REPETITION_PENALTY = float(os.getenv("YUE_REPETITION_PENALTY", "1.1"))
DEFAULT_STAGE2_BATCH_SIZE = int(os.getenv("YUE_STAGE2_BATCH_SIZE", "4"))
DEFAULT_RUN_N_SEGMENTS = int(os.getenv("YUE_RUN_N_SEGMENTS", "2"))
DEFAULT_SEED = int(os.getenv("YUE_SEED", "42"))

# Example prompts
EXAMPLE_GENRE = "inspiring female uplifting pop airy vocal electronic bright vocal"
EXAMPLE_LYRICS = """[verse]
Running in the night
My heart beats like a drum
I'm searching for the light
In this city so cold

[chorus]
We are the dreamers
Fighting through the dark
We are believers
Following our heart"""

def log_reader(process):
    """Read process output and put it in the log queue."""
    try:
        for line in iter(process.stdout.readline, b''):
            if line:
                log_queue.put(line.decode('utf-8', errors='ignore'))
        for line in iter(process.stderr.readline, b''):
            if line:
                log_queue.put(line.decode('utf-8', errors='ignore'))
    except Exception as e:
        log_queue.put(f"Log reader error: {str(e)}\n")

def stop_generation():
    """Stop the current generation process."""
    global generation_process
    with process_lock:
        if generation_process and generation_process.poll() is None:
            try:
                generation_process.terminate()
                generation_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                generation_process.kill()
                generation_process.wait()
            generation_process = None
            return "Generation stopped successfully."
        else:
            return "No active generation process to stop."

def generate_music(
    stage1_model,
    stage2_model,
    genre_text,
    lyrics_text,
    max_new_tokens,
    repetition_penalty,
    stage2_batch_size,
    run_n_segments,
    seed,
    use_audio_prompt,
    audio_prompt_file,
    prompt_start_time,
    prompt_end_time,
    use_dual_tracks_prompt,
    vocal_track_file,
    instrumental_track_file,
    cuda_idx,
    keep_intermediate,
    progress=gr.Progress()
):
    """Generate music using YuE with the provided parameters."""
    global generation_process
    
    with process_lock:
        if generation_process and generation_process.poll() is None:
            return "Another generation is already running. Please stop it first.", None, gr.Button(interactive=True), gr.Button(interactive=False)
    
    try:
        progress(0.1, desc="Preparing generation...")
        
        # Create temporary files for genre and lyrics
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(genre_text)
            genre_file = f.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(lyrics_text)
            lyrics_file = f.name
        
        # Build command
        cmd = [
            "conda", "run", "-n", "yue", "python", "-u", 
            os.path.join(YUE_PATH, "inference", "infer.py"),
            "--stage1_model", stage1_model,
            "--stage2_model", stage2_model,
            "--genre_txt_path", genre_file,
            "--lyrics_txt_path", lyrics_file,
            "--max_new_tokens", str(max_new_tokens),
            "--repetition_penalty", str(repetition_penalty),
            "--stage2_batch_size", str(stage2_batch_size),
            "--run_n_segments", str(run_n_segments),
            "--cuda_idx", str(cuda_idx),
            "--seed", str(seed),
            "--output_dir", OUTPUT_DIR
        ]
        
        if keep_intermediate:
            cmd.append("--keep_intermediate")
        
        # Handle audio prompts
        if use_audio_prompt and audio_prompt_file:
            cmd.extend(["--audio_prompt_path", audio_prompt_file.name])
            cmd.extend(["--prompt_start_time", str(prompt_start_time)])
            cmd.extend(["--prompt_end_time", str(prompt_end_time)])
        
        if use_dual_tracks_prompt and vocal_track_file and instrumental_track_file:
            cmd.extend(["--vocal_track_prompt_path", vocal_track_file.name])
            cmd.extend(["--instrumental_track_prompt_path", instrumental_track_file.name])
        
        progress(0.2, desc="Starting generation process...")
        
        # Start process
        with process_lock:
            generation_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=YUE_PATH,
                env=dict(os.environ, 
                    CUDA_VISIBLE_DEVICES=str(cuda_idx),
                    HF_HOME=CACHE_DIR + "/huggingface",
                    TORCH_HOME=CACHE_DIR + "/torch"
                )
            )
        
        # Start log reader thread
        log_thread = threading.Thread(target=log_reader, args=(generation_process,), daemon=True)
        log_thread.start()
        
        progress(0.3, desc="Generation in progress...")
        
        # Cleanup temp files
        try:
            os.unlink(genre_file)
            os.unlink(lyrics_file)
        except:
            pass
        
        return "Generation started successfully!", generation_process.pid, gr.Button(interactive=False), gr.Button(interactive=True)
        
    except Exception as e:
        # Cleanup temp files on error
        try:
            if 'genre_file' in locals():
                os.unlink(genre_file)
            if 'lyrics_file' in locals():
                os.unlink(lyrics_file)
        except:
            pass
        
        return f"Error starting generation: {str(e)}", None, gr.Button(interactive=True), gr.Button(interactive=False)

def get_generation_logs():
    """Get the latest logs from the generation process."""
    logs = ""
    while not log_queue.empty():
        try:
            logs += log_queue.get_nowait()
        except queue.Empty:
            break
    return logs

def list_output_files():
    """List generated output files."""
    output_path = Path(OUTPUT_DIR)
    if not output_path.exists():
        return []
    
    # Look for audio files
    audio_files = []
    for ext in ['*.wav', '*.mp3', '*.flac']:
        audio_files.extend(output_path.glob(f"**/{ext}"))
    
    # Sort by modification time (newest first)
    audio_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    return [str(f) for f in audio_files[:10]]  # Return only the 10 most recent files

def load_example_prompts():
    """Load example prompts from the YuE repository."""
    examples = {}
    
    # Try to load genre examples
    genre_path = Path(YUE_PATH) / "prompt_egs" / "genre.txt"
    if genre_path.exists():
        with open(genre_path, 'r') as f:
            examples['genre'] = f.read().strip()
    else:
        examples['genre'] = EXAMPLE_GENRE
    
    # Try to load lyrics examples  
    lyrics_path = Path(YUE_PATH) / "prompt_egs" / "lyrics.txt"
    if lyrics_path.exists():
        with open(lyrics_path, 'r') as f:
            examples['lyrics'] = f.read().strip()
    else:
        examples['lyrics'] = EXAMPLE_LYRICS
    
    return examples

def create_interface():
    """Create the Gradio interface."""
    
    # Load examples
    examples = load_example_prompts()
    
    with gr.Blocks(
        title="YuE Music Generation",
        theme=gr.themes.Soft(),
        css="""
        .container { max-width: 1200px; margin: auto; }
        .header { text-align: center; padding: 20px; }
        .examples { background-color: #f8f9fa; padding: 15px; border-radius: 8px; margin: 10px 0; }
        """
    ) as interface:
        
        gr.HTML("""
        <div class="header">
            <h1>🎵 YuE Music Generation Interface</h1>
            <p>Generate high-quality music from text prompts using the YuE model</p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=2):
                # Model Configuration
                with gr.Group():
                    gr.Markdown("### 🎯 Model Configuration")
                    stage1_model = gr.Textbox(
                        label="Stage 1 Model",
                        value=DEFAULT_STAGE1_MODEL,
                        info="HuggingFace model ID or local path for Stage 1 (text-to-semantic tokens)"
                    )
                    stage2_model = gr.Textbox(
                        label="Stage 2 Model", 
                        value=DEFAULT_STAGE2_MODEL,
                        info="HuggingFace model ID or local path for Stage 2 (semantic-to-audio tokens)"
                    )
                
                # Generation Parameters
                with gr.Group():
                    gr.Markdown("### ⚙️ Generation Parameters")
                    max_new_tokens = gr.Slider(
                        label="Max New Tokens",
                        minimum=500,
                        maximum=6000,
                        step=100,
                        value=DEFAULT_MAX_NEW_TOKENS,
                        info="Maximum number of new tokens to generate (affects duration)"
                    )
                    repetition_penalty = gr.Slider(
                        label="Repetition Penalty",
                        minimum=1.0,
                        maximum=2.0,
                        step=0.05,
                        value=DEFAULT_REPETITION_PENALTY,
                        info="Penalty for repeated patterns (1.0 = no penalty)"
                    )
                    stage2_batch_size = gr.Slider(
                        label="Stage 2 Batch Size",
                        minimum=1,
                        maximum=16,
                        step=1,
                        value=DEFAULT_STAGE2_BATCH_SIZE,
                        info="Batch size for Stage 2 processing (higher = faster but more VRAM)"
                    )
                    run_n_segments = gr.Slider(
                        label="Number of Segments",
                        minimum=1,
                        maximum=10,
                        step=1,
                        value=DEFAULT_RUN_N_SEGMENTS,
                        info="Number of lyric segments to generate"
                    )
                    seed = gr.Number(
                        label="Seed",
                        value=DEFAULT_SEED,
                        precision=0,
                        info="Random seed for reproducible generation"
                    )
                    cuda_idx = gr.Number(
                        label="CUDA Device Index",
                        value=DEFAULT_CUDA_IDX,
                        precision=0,
                        info="GPU device index to use"
                    )
                    keep_intermediate = gr.Checkbox(
                        label="Keep Intermediate Files",
                        value=True,
                        info="Save intermediate generation stages"
                    )
            
            with gr.Column(scale=3):
                # Text Prompts
                with gr.Group():
                    gr.Markdown("### 📝 Text Prompts")
                    
                    gr.HTML('<div class="examples">💡 <strong>Genre Tips:</strong> Include 5 components: genre, instrument, mood, vocal gender, and vocal timbre (e.g., "bright vocal")</div>')
                    
                    genre_text = gr.Textbox(
                        label="Genre Description",
                        value=examples.get('genre', EXAMPLE_GENRE),
                        lines=2,
                        info="Describe the musical style, instruments, mood, and vocal characteristics"
                    )
                    
                    gr.HTML('<div class="examples">💡 <strong>Lyrics Tips:</strong> Structure lyrics with sections like [verse], [chorus], [bridge]. Each section should be around 15-30 seconds worth of content.</div>')
                    
                    lyrics_text = gr.Textbox(
                        label="Lyrics",
                        value=examples.get('lyrics', EXAMPLE_LYRICS),
                        lines=8,
                        info="Structured lyrics with sections marked by [section_name]"
                    )
                
                # Audio Prompts (ICL - In-Context Learning)
                with gr.Group():
                    gr.Markdown("### 🎧 Audio Prompts (Optional - In-Context Learning)")
                    
                    use_audio_prompt = gr.Checkbox(
                        label="Use Single Audio Prompt",
                        value=False,
                        info="Use an audio file as reference for style"
                    )
                    
                    with gr.Row():
                        audio_prompt_file = gr.File(
                            label="Audio Prompt File",
                            file_types=["audio"],
                            visible=False
                        )
                        with gr.Column():
                            prompt_start_time = gr.Number(
                                label="Start Time (s)",
                                value=0,
                                visible=False
                            )
                            prompt_end_time = gr.Number(
                                label="End Time (s)",
                                value=30,
                                visible=False
                            )
                    
                    use_dual_tracks_prompt = gr.Checkbox(
                        label="Use Dual Track Prompts (Recommended)",
                        value=False,
                        info="Use separate vocal and instrumental tracks for better quality"
                    )
                    
                    with gr.Row():
                        vocal_track_file = gr.File(
                            label="Vocal Track",
                            file_types=["audio"],
                            visible=False
                        )
                        instrumental_track_file = gr.File(
                            label="Instrumental Track", 
                            file_types=["audio"],
                            visible=False
                        )
                
                # Generation Controls
                with gr.Group():
                    gr.Markdown("### 🚀 Generation")
                    with gr.Row():
                        generate_btn = gr.Button("🎵 Generate Music", variant="primary", scale=2)
                        stop_btn = gr.Button("⏹️ Stop", variant="stop", scale=1, interactive=False)
                    
                    status_text = gr.Textbox(
                        label="Status",
                        value="Ready to generate",
                        interactive=False
                    )
        
        # Results Section
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 📊 Generation Logs")
                logs_text = gr.Textbox(
                    label="Logs",
                    lines=15,
                    max_lines=30,
                    interactive=False,
                    show_copy_button=True
                )
                
                # Auto-refresh logs
                logs_timer = gr.Timer(value=2.0, active=False)
                
            with gr.Column():
                gr.Markdown("### 🎵 Generated Music")
                
                output_files = gr.Dropdown(
                    label="Generated Files",
                    choices=[],
                    interactive=True,
                    info="Select a generated audio file to play"
                )
                
                audio_player = gr.Audio(
                    label="Audio Player",
                    type="filepath"
                )
                
                refresh_files_btn = gr.Button("🔄 Refresh Files")
        
        # Hidden state for process ID
        process_id = gr.State(None)
        
        # Event handlers
        def toggle_audio_prompt_visibility(use_audio):
            return [
                gr.File(visible=use_audio),
                gr.Number(visible=use_audio),
                gr.Number(visible=use_audio)
            ]
        
        def toggle_dual_track_visibility(use_dual):
            return [
                gr.File(visible=use_dual),
                gr.File(visible=use_dual)
            ]
        
        def handle_audio_prompt_change(use_audio, use_dual):
            if use_audio:
                return False, *toggle_audio_prompt_visibility(True), *toggle_dual_track_visibility(False)
            return use_dual, *toggle_audio_prompt_visibility(False), *toggle_dual_track_visibility(use_dual)
        
        def handle_dual_track_change(use_audio, use_dual):
            if use_dual:
                return False, *toggle_audio_prompt_visibility(False), *toggle_dual_track_visibility(True)
            return use_audio, *toggle_audio_prompt_visibility(use_audio), *toggle_dual_track_visibility(False)
        
        # Wire up visibility toggles
        use_audio_prompt.change(
            fn=handle_audio_prompt_change,
            inputs=[use_audio_prompt, use_dual_tracks_prompt],
            outputs=[use_dual_tracks_prompt, audio_prompt_file, prompt_start_time, prompt_end_time, vocal_track_file, instrumental_track_file]
        )
        
        use_dual_tracks_prompt.change(
            fn=handle_dual_track_change,
            inputs=[use_audio_prompt, use_dual_tracks_prompt],
            outputs=[use_audio_prompt, audio_prompt_file, prompt_start_time, prompt_end_time, vocal_track_file, instrumental_track_file]
        )
        
        # Generation event
        generate_btn.click(
            fn=generate_music,
            inputs=[
                stage1_model, stage2_model, genre_text, lyrics_text,
                max_new_tokens, repetition_penalty, stage2_batch_size, run_n_segments,
                seed, use_audio_prompt, audio_prompt_file, prompt_start_time, prompt_end_time,
                use_dual_tracks_prompt, vocal_track_file, instrumental_track_file,
                cuda_idx, keep_intermediate
            ],
            outputs=[status_text, process_id, generate_btn, stop_btn]
        ).then(
            fn=lambda: gr.Timer(active=True),
            outputs=[logs_timer]
        )
        
        # Stop event
        stop_btn.click(
            fn=stop_generation,
            outputs=[status_text]
        ).then(
            fn=lambda: [gr.Button(interactive=True), gr.Button(interactive=False), gr.Timer(active=False)],
            outputs=[generate_btn, stop_btn, logs_timer]
        )
        
        # Log updates
        def update_logs(current_logs):
            new_logs = get_generation_logs()
            if new_logs:
                return current_logs + new_logs
            return gr.update()
        
        logs_timer.tick(
            fn=update_logs,
            inputs=[logs_text],
            outputs=[logs_text]
        )
        
        # File refresh
        def refresh_output_files():
            files = list_output_files()
            choices = [(os.path.basename(f), f) for f in files]
            return gr.Dropdown(choices=choices)
        
        refresh_files_btn.click(
            fn=refresh_output_files,
            outputs=[output_files]
        )
        
        # File selection
        def load_audio_file(selected_file):
            if selected_file:
                return selected_file
            return None
        
        output_files.change(
            fn=load_audio_file,
            inputs=[output_files],
            outputs=[audio_player]
        )
        
        # Auto-refresh files on interface load
        interface.load(
            fn=refresh_output_files,
            outputs=[output_files]
        )
    
    return interface

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="YuE Gradio Interface")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=7860, help="Port to bind the server to")
    parser.add_argument("--share", action="store_true", help="Create a public link")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Create and launch interface
    interface = create_interface()
    
    print(f"🎵 Starting YuE Gradio Interface on http://{args.host}:{args.port}")
    print(f"📁 Output directory: {OUTPUT_DIR}")
    print(f"💾 Cache directory: {CACHE_DIR}")
    
    interface.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        debug=args.debug,
        show_error=True,
        quiet=False
    )
