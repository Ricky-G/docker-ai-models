"""
YuE GPU-Poor Gradio Interface
RTX 3060 12GB optimized with mmgp Profile 3
"""
import os
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

import gc
import sys
import time
from datetime import datetime
import torch
import gradio as gr
import numpy as np
import soundfile as sf
from pathlib import Path
from transformers import AutoModelForCausalLM, LogitsProcessor, LogitsProcessorList
from huggingface_hub import snapshot_download
from mmgp import offload
from einops import rearrange

# Global state
STATE = {"device": None, "models": {}, "tokenizer": None, "codec": None}

def log(msg):
    """Print message with timestamp"""
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")

def initialize_models(profile=3):
    """Initialize YuE models with mmgp optimization"""
    print("🔧 Initializing YuE models...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    STATE["device"] = device
    
    # Get HuggingFace token from environment
    hf_token = os.environ.get("HF_TOKEN", None)
    
    # Download xcodec from HuggingFace to local directory
    print("   Downloading xcodec (first time only)...")
    xcodec_local = Path("/app/xcodec_mini_infer")
    if not xcodec_local.exists():
        snapshot_download(
            repo_id="m-a-p/xcodec_mini_infer",
            token=hf_token,
            local_dir=str(xcodec_local),
            local_dir_use_symlinks=False
        )
    
    # Download tokenizer and inference code from YuEGP GitHub repo
    print("   Downloading tokenizer and inference code (first time only)...")
    inference_dir = Path("/app/YuEGP_inference")
    if not inference_dir.exists():
        import subprocess
        subprocess.run([
            "git", "clone", "--depth=1", 
            "https://github.com/deepbeepmeep/YuEGP.git",
            "/tmp/YuEGP"
        ], check=True)
        import shutil
        # Copy inference directory which has codecmanipulator, mmtokenizer, etc.
        shutil.copytree("/tmp/YuEGP/inference", inference_dir)
        shutil.rmtree("/tmp/YuEGP")
    
    # Add xcodec and inference to Python path
    sys.path.insert(0, str(xcodec_local))
    sys.path.insert(0, str(xcodec_local / "descriptaudiocodec"))
    sys.path.insert(0, str(inference_dir))
    
    # Import xcodec modules after setting up paths
    from codecmanipulator import CodecManipulator
    from mmtokenizer import _MMSentencePieceTokenizer
    from omegaconf import OmegaConf
    from models.soundstream_hubert_new import SoundStream
    from vocoder import build_codec_model
    
    # Load tokenizer
    tokenizer_path = inference_dir / "mm_tokenizer_v0.2_hf" / "tokenizer.model"
    STATE["tokenizer"] = _MMSentencePieceTokenizer(str(tokenizer_path))
    
    # Load Stage 1 model (lyrics to semantic tokens)
    print("   Loading Stage 1 model (7B)...")
    model_s1 = AutoModelForCausalLM.from_pretrained(
        "m-a-p/YuE-s1-7B-anneal-en-cot",
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",  # Fallback to SDPA if flash-attn unavailable
        token=hf_token,
    )
    model_s1.to("cpu")
    model_s1.eval()
    model_s1._validate_model_kwargs = lambda x: None  # Disable validation
    
    # Load Stage 2 model (semantic to audio tokens)
    print("   Loading Stage 2 model (1B)...")
    model_s2 = AutoModelForCausalLM.from_pretrained(
        "m-a-p/YuE-s2-1B-general",
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
        token=hf_token,
    )
    model_s2.to("cpu")
    model_s2.eval()
    model_s2._validate_model_kwargs = lambda x: None
    
    STATE["models"] = {"stage1": model_s1, "stage2": model_s2}
    
    # Apply mmgp Profile 3 optimization
    print(f"   Applying mmgp Profile {profile} (12GB VRAM + 8-bit quantization)...")
    pipe = {"transformer": model_s1, "stage2": model_s2}
    quantize = profile >= 3  # Profiles 3-5 use quantization
    offload.profile(pipe, profile_no=profile, quantizeTransformer=quantize, verboseLevel=1)
    
    # Load xcodec for audio encoding/decoding
    print("   Loading xcodec...")
    xcodec_config = xcodec_local / "final_ckpt" / "config.yaml"
    xcodec_ckpt = xcodec_local / "final_ckpt" / "ckpt_00360000.pth"
    
    model_config = OmegaConf.load(str(xcodec_config))
    codec_model = eval(model_config.generator.name)(**model_config.generator.config).to(device)
    checkpoint = torch.load(str(xcodec_ckpt), map_location="cpu")
    codec_model.load_state_dict(checkpoint["codec_model"])
    codec_model.eval()
    STATE["codec"] = codec_model
    
    # Codec manipulators
    STATE["codec_tool"] = CodecManipulator("xcodec", 0, 1)
    STATE["codec_tool_s2"] = CodecManipulator("xcodec", 0, 8)
    
    # Vocoder for upsampling
    vocal_decoder = xcodec_local / "decoders" / "decoder_131000.pth"
    inst_decoder = xcodec_local / "decoders" / "decoder_151000.pth"
    vocoder_config = xcodec_local / "decoders" / "config.yaml"
    STATE["vocoder"] = build_codec_model(str(vocoder_config), str(vocal_decoder), str(inst_decoder))
    
    print("✅ Models loaded and optimized")


class BlockTokenRangeProcessor(LogitsProcessor):
    def __init__(self, start_id, end_id):
        self.blocked = list(range(start_id, end_id))
    
    def __call__(self, input_ids, scores):
        scores[:, self.blocked] = -float("inf")
        return scores


def split_lyrics(lyrics):
    """Split lyrics into sections"""
    sections = []
    current = []
    for line in lyrics.strip().split("\n"):
        if line.strip().startswith("[") and current:
            sections.append("\n".join(current))
            current = [line]
        else:
            current.append(line)
    if current:
        sections.append("\n".join(current))
    return sections


def generate_song(genres, lyrics, num_segments=2, seed=42):
    """Generate song from lyrics and genre tags"""
    if not STATE["models"]:
        return "⚠️ Models not loaded. Please wait for initialization.", None, None
    
    start_time = time.time()
    log(f"🎵 Generating song (seed={seed}, segments={num_segments})...")
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Clear CUDA cache to ensure clean state for each run
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    mmtokenizer = STATE["tokenizer"]
    model_s1 = STATE["models"]["stage1"]
    model_s2 = STATE["models"]["stage2"]
    device = STATE["device"]
    codec_tool = STATE["codec_tool"]
    
    # Stage 1: Lyrics to semantic tokens
    stage1_start = time.time()
    log("   Stage 1: Lyrics → Semantic tokens...")
    
    # Force model to GPU before starting (mmgp may have it offloaded to CPU)
    # This ensures logits processors work correctly on the first segment
    model_device = next(model_s1.parameters()).device
    if model_device != device:
        log(f"   Moving Stage 1 model from {model_device} to {device}...")
        model_s1.to(device)
        torch.cuda.empty_cache()
    
    lyrics_sections = split_lyrics(lyrics)
    full_lyrics = "\n".join(lyrics_sections)
    
    prompt_texts = [f"Generate music from the given lyrics segment by segment.\n[Genre] {genres}\n{full_lyrics}"]
    prompt_texts += lyrics_sections
    
    # Special tokens
    start_of_segment = mmtokenizer.tokenize('[start_of_segment]')
    end_of_segment = mmtokenizer.tokenize('[end_of_segment]')
    
    raw_output = None
    run_segments = min(num_segments + 1, len(prompt_texts))
    
    for i in range(run_segments):
        section_text = prompt_texts[i].replace('[start_of_segment]', '').replace('[end_of_segment]', '')
        
        if i == 0:
            # First iteration - just the header
            head_id = mmtokenizer.tokenize(section_text)
            continue
        
        if i == 1:
            # First segment
            prompt_ids = head_id + start_of_segment + mmtokenizer.tokenize(section_text) + [mmtokenizer.soa] + codec_tool.sep_ids
        else:
            # Subsequent segments
            prompt_ids = end_of_segment + start_of_segment + mmtokenizer.tokenize(section_text) + [mmtokenizer.soa] + codec_tool.sep_ids
        
        # Tensors should be on CUDA for model.generate() to work properly
        prompt_ids = torch.as_tensor(prompt_ids).unsqueeze(0).to(device)
        input_ids = torch.cat([raw_output, prompt_ids], dim=1) if raw_output is not None else prompt_ids
        
        # DEBUG: Print device info
        log(f"   DEBUG: device={device}, input_ids.device={input_ids.device}")
        log(f"   DEBUG: model device check - next(model_s1.parameters()).device={next(model_s1.parameters()).device}")
        
        # Limit context length
        max_context = 16384 - 3000 - 1
        if input_ids.shape[-1] > max_context:
            log(f"   Segment {i}: Using last {max_context} tokens (context overflow)")
            input_ids = input_ids[:, -max_context:]
        
        segment_start = time.time()
        log(f"   DEBUG: Starting generate for segment {i}...")
        with torch.no_grad():
            raw_output = model_s1.generate(
                input_ids,
                max_new_tokens=3000,
                min_new_tokens=100,
                do_sample=True,
                top_p=0.93,
                temperature=1.0,
                repetition_penalty=1.2,
                eos_token_id=mmtokenizer.eoa,
                pad_token_id=mmtokenizer.eoa,
                logits_processor=LogitsProcessorList([
                    BlockTokenRangeProcessor(0, 32002),
                    BlockTokenRangeProcessor(32016, 32017)  # Fixed: was 32016, 32016
                ]),
            )
        segment_time = time.time() - segment_start
        log(f"   DEBUG: Generate completed. raw_output shape={raw_output.shape}, device={raw_output.device} ({segment_time:.1f}s)")
    
    # Extract vocals and instrumentals
    ids = raw_output[0].cpu().numpy()
    soa_idx = np.where(ids == mmtokenizer.soa)[0]
    eoa_idx = np.where(ids == mmtokenizer.eoa)[0]
    
    vocals, instrumentals = [], []
    
    for i in range(len(soa_idx)):
        start = soa_idx[i] + 1
        end = eoa_idx[i] if i < len(eoa_idx) else len(ids)
        codec_ids = ids[start:end]
        
        if len(codec_ids) == 0:
            continue
            
        # Skip separator token if present
        if codec_ids[0] == 32016:
            codec_ids = codec_ids[1:]
        
        # Ensure even length
        codec_ids = codec_ids[:2 * (len(codec_ids) // 2)]
        if len(codec_ids) == 0:
            continue
        
        # Validate codec IDs are in correct range (should be >= 45334 for audio tokens)
        min_id = codec_ids.min()
        max_id = codec_ids.max()
        log(f"   DEBUG: Segment {i} codec_ids: len={len(codec_ids)}, min={min_id}, max={max_id}")
        if min_id < 45334:
            log(f"   ⚠️ WARNING: Found text tokens in output (min={min_id}, expected >=45334)")
            log(f"   This may indicate the model generated text instead of audio tokens")
            # Filter out invalid tokens (keep only codec tokens >= 45334)
            valid_mask = codec_ids >= 45334
            if valid_mask.sum() < 2:
                log(f"   ❌ Segment {i} has no valid codec tokens, skipping")
                continue
            codec_ids = codec_ids[valid_mask]
            codec_ids = codec_ids[:2 * (len(codec_ids) // 2)]  # Re-ensure even length
            log(f"   Filtered to {len(codec_ids)} valid codec tokens")
        
        vocal_ids = codec_tool.ids2npy(rearrange(codec_ids, "(n b) -> b n", b=2)[0])
        inst_ids = codec_tool.ids2npy(rearrange(codec_ids, "(n b) -> b n", b=2)[1])
        vocals.append(vocal_ids)
        instrumentals.append(inst_ids)
    
    vocals = np.concatenate(vocals, axis=1)
    instrumentals = np.concatenate(instrumentals, axis=1)
    
    stage1_time = time.time() - stage1_start
    log(f"   Stage 1 completed in {stage1_time:.1f}s")
    
    # Offload Stage 1 model back to CPU to free VRAM for Stage 2
    log("   Offloading Stage 1 model to free VRAM...")
    model_s1.to("cpu")
    torch.cuda.empty_cache()
    gc.collect()
    log(f"   After Stage 1 offload - GPU memory: {torch.cuda.memory_allocated()/1024**3:.1f}GB")
    
    # Also offload codec model - Stage 2 doesn't need it until audio decoding
    codec_model = STATE["codec"]
    log("   Offloading codec model...")
    codec_model.to("cpu")
    torch.cuda.empty_cache()
    gc.collect()
    log(f"   After codec offload - GPU memory: {torch.cuda.memory_allocated()/1024**3:.1f}GB")
    
    # Stage 2: Semantic → Audio tokens
    stage2_start = time.time()
    log("   Stage 2: Semantic → Audio tokens...")
    
    vocal_audio = stage2_inference(vocals, "vocal")
    inst_audio = stage2_inference(instrumentals, "instrumental")
    
    # Mix tracks
    mix_audio = (vocal_audio + inst_audio) / 1.0
    
    # Save outputs
    output_dir = Path("/app/output")
    output_dir.mkdir(exist_ok=True)
    
    vocal_path = output_dir / "vocal.wav"
    inst_path = output_dir / "instrumental.wav"
    mix_path = output_dir / "mix.wav"
    
    sf.write(str(vocal_path), vocal_audio, 44100)
    sf.write(str(inst_path), inst_audio, 44100)
    sf.write(str(mix_path), mix_audio, 44100)
    
    stage2_time = time.time() - stage2_start
    total_time = time.time() - start_time
    log(f"   Stage 2 completed in {stage2_time:.1f}s")
    log(f"✅ Generation complete! Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    return f"✅ Song generated successfully! ({total_time:.1f}s)", str(mix_path), str(vocal_path)


def stage2_inference(semantic_npy, track_type):
    """Convert semantic tokens (from ids2npy) to audio using Stage 2 model
    
    Reference implementation from YuEGP/inference/infer.py
    mmgp handles model layer offloading automatically - don't force to GPU!
    """
    model_s2 = STATE["models"]["stage2"]
    codec_tool = STATE["codec_tool"]
    codec_tool_s2 = STATE["codec_tool_s2"]
    tokenizer = STATE["tokenizer"]
    device = STATE["device"]
    vocoder = STATE["vocoder"]
    codec_model = STATE["codec"]  # soundstream model
    
    # DON'T force Stage 2 model to GPU - mmgp handles layer-by-layer offloading
    # Moving the entire model defeats mmgp's memory management
    model_device = next(model_s2.parameters()).device
    log(f"      Stage 2 model device: {model_device} (mmgp managed)")
    
    # semantic_npy is (1, T) from ids2npy - values are 0-1023
    prompt = semantic_npy.astype(np.int32)
    log(f"      Stage2 input shape: {prompt.shape}, min={prompt.min()}, max={prompt.max()}")
    
    # Calculate output duration (6 second segments = 300 frames at 50fps)
    total_frames = prompt.shape[1]
    output_duration = (total_frames // 50 // 6) * 6  # In seconds, 6s chunks
    num_segments = output_duration // 6
    
    log(f"      Total frames: {total_frames}, output_duration: {output_duration}s, segments: {num_segments}")
    
    all_outputs = []
    
    if num_segments == 0:
        # Less than 6 seconds, process all
        log(f"      Processing {total_frames} frames as single batch...")
        output = stage2_generate(prompt, tokenizer, model_s2, device, codec_tool)
        all_outputs.append(output)
    else:
        # Process in 6-second segments (300 frames each)
        segment_frames = 300
        for seg in range(num_segments):
            start_idx = seg * segment_frames
            end_idx = (seg + 1) * segment_frames
            segment_prompt = prompt[:, start_idx:end_idx]
            log(f"      Processing segment {seg+1}/{num_segments} (frames {start_idx}-{end_idx})...")
            segment_output = stage2_generate(segment_prompt, tokenizer, model_s2, device, codec_tool)
            all_outputs.append(segment_output)
            
            # Critical: Clear VRAM between segments to prevent accumulation
            torch.cuda.empty_cache()
            gc.collect()
            vram_used = torch.cuda.memory_allocated() / 1024**3
            log(f"      Segment {seg+1} done. VRAM: {vram_used:.1f}GB")
        
        # Handle remaining frames
        remaining_start = output_duration * 50
        if remaining_start < total_frames:
            remaining_prompt = prompt[:, remaining_start:]
            if remaining_prompt.shape[1] > 0:
                # Extra cleanup before the final segment
                torch.cuda.empty_cache()
                gc.collect()
                log(f"      Processing remaining {remaining_prompt.shape[1]} frames... (VRAM: {torch.cuda.memory_allocated()/1024**3:.1f}GB)")
                remaining_output = stage2_generate(remaining_prompt, tokenizer, model_s2, device, codec_tool)
                all_outputs.append(remaining_output)
    
    output = np.concatenate(all_outputs, axis=0)
    log(f"      Stage2 output: {output.shape}, min={output.min()}, max={output.max()}")
    
    # Convert stage2 output to codec format (8 codebooks)
    output_npy = codec_tool_s2.ids2npy(output)
    log(f"      Codec npy shape: {output_npy.shape}")
    
    # Fix invalid codes (values outside 0-1023 range)
    import copy
    from collections import Counter
    fixed_output = copy.deepcopy(output_npy)
    invalid_count = 0
    for i, line in enumerate(output_npy):
        for j, element in enumerate(line):
            if element < 0 or element > 1023:
                invalid_count += 1
                counter = Counter(line)
                most_frequent = sorted(counter.items(), key=lambda x: x[1], reverse=True)[0][0]
                fixed_output[i, j] = most_frequent
    if invalid_count > 0:
        log(f"      Fixed {invalid_count} invalid codes")
    
    # Clear VRAM before audio decode by forcing mmgp to unload all model blocks
    log(f"      Clearing VRAM before audio decode... (VRAM: {torch.cuda.memory_allocated()/1024**3:.1f}GB)")
    
    # Access mmgp's global offload object and force unload all blocks
    import mmgp.offload as mmgp_offload
    if hasattr(mmgp_offload, 'last_offload_obj') and mmgp_offload.last_offload_obj is not None:
        log(f"      Forcing mmgp to unload all model blocks from GPU...")
        mmgp_offload.last_offload_obj.unload_all()
    
    torch.cuda.empty_cache()
    gc.collect()
    log(f"      After mmgp unload - VRAM: {torch.cuda.memory_allocated()/1024**3:.1f}GB")
    
    # Reload codec model to GPU for audio decoding
    log(f"      Reloading codec model to GPU for audio decode...")
    codec_model.to(device)
    torch.cuda.empty_cache()
    log(f"      Codec model loaded - VRAM: {torch.cuda.memory_allocated()/1024**3:.1f}GB")
    
    # First decode at 16kHz using codec_model.decode (matching reference)
    # Shape needs to be (8, 1, T) for the codec decoder
    # Use chunked decoding for long sequences to avoid OOM
    with torch.no_grad():
        codec_input = torch.as_tensor(fixed_output.astype(np.int16), dtype=torch.long).unsqueeze(0).permute(1, 0, 2).to(device)
        log(f"      Codec input shape for decode: {codec_input.shape}")  # Should be (8, 1, T)
        
        # Chunk the decode if sequence is too long
        T = codec_input.shape[2]
        chunk_size = 500  # Process 500 frames at a time
        if T > chunk_size:
            log(f"      Using chunked decode ({T} frames in {(T + chunk_size - 1) // chunk_size} chunks)...")
            decoded_chunks = []
            for i in range(0, T, chunk_size):
                end_idx = min(i + chunk_size, T)
                chunk = codec_input[:, :, i:end_idx]
                decoded_chunk = codec_model.decode(chunk)
                decoded_chunks.append(decoded_chunk.cpu())
                torch.cuda.empty_cache()
            decoded_waveform = torch.cat(decoded_chunks, dim=-1).squeeze(0)
        else:
            decoded_waveform = codec_model.decode(codec_input)
            decoded_waveform = decoded_waveform.cpu().squeeze(0)
        log(f"      Decoded waveform shape: {decoded_waveform.shape}, sr=16000")
    
    # Now upsample to 44.1kHz using vocoder
    vocal_decoder, inst_decoder = vocoder
    decoder = vocal_decoder if track_type == "vocal" else inst_decoder
    decoder.eval()
    decoder = decoder.to(device)
    
    with torch.no_grad():
        # Get embeddings and decode with vocoder for 44.1kHz output
        codec_embed = torch.as_tensor(fixed_output.astype(np.int16), dtype=torch.long).unsqueeze(1).to(device)
        embeddings = codec_model.get_embed(codec_embed)
        embeddings = torch.tensor(embeddings).to(device)
        log(f"      Vocoder input embeddings shape: {embeddings.shape}")
        
        # Chunk vocoder decode if needed
        T = embeddings.shape[0]
        chunk_size = 500  # Process 500 frames at a time
        if T > chunk_size:
            log(f"      Using chunked vocoder decode ({T} frames)...")
            audio_chunks = []
            for i in range(0, T, chunk_size):
                end_idx = min(i + chunk_size, T)
                chunk = embeddings[i:end_idx]
                audio_chunk = decoder(chunk)
                audio_chunks.append(audio_chunk.squeeze().detach().cpu())
                torch.cuda.empty_cache()
            audio = torch.cat(audio_chunks, dim=-1).numpy()
        else:
            audio_44k = decoder(embeddings)
            audio = audio_44k.squeeze().detach().cpu().numpy()
        log(f"      Final audio shape: {audio.shape}, sample_rate=44100")
    
    # CRITICAL: Offload codec model and vocoder after audio decode
    # This prepares GPU for the next track (instrumental) or frees memory
    log(f"      Cleaning up after {track_type} audio decode...")
    codec_model.to("cpu")
    decoder.to("cpu")
    torch.cuda.empty_cache()
    gc.collect()
    log(f"      After cleanup - VRAM: {torch.cuda.memory_allocated()/1024**3:.1f}GB")
    
    return audio


def stage2_generate(prompt, tokenizer, model_s2, device, codec_tool):
    """Generate 8-codebook audio tokens from 1-codebook semantic tokens
    
    Args:
        prompt: numpy array (1, T) with values 0-1023 (raw codec values from ids2npy)
        tokenizer: MMTokenizer
        model_s2: Stage 2 model
        device: torch device
        codec_tool: CodecManipulator for xcodec (n_quantizer=1)
    """
    # Unflatten prompt to get (1, T) codec_ids and add offset
    codec_ids = codec_tool.unflatten(prompt, n_quantizer=1)
    codec_ids = codec_tool.offset_tok_ids(
        codec_ids,
        global_offset=codec_tool.global_offset,
        codebook_size=codec_tool.codebook_size,
        num_codebooks=codec_tool.num_codebooks,
    ).astype(np.int32)
    
    # Build initial prompt: [soa, stage_1, codec_ids_flat, stage_2]
    prompt_ids = np.concatenate([
        np.array([tokenizer.soa, tokenizer.stage_1]),
        codec_ids.flatten(),
        np.array([tokenizer.stage_2])
    ]).astype(np.int32)
    # IMPORTANT: Put tensors on CUDA - mmgp expects CUDA inputs even though it manages offloading
    # The embedding layer stays on CUDA, so input_ids must be on CUDA too
    prompt_ids = torch.as_tensor(prompt_ids).unsqueeze(0).to(device)
    
    # Convert codec_ids to tensor for indexing (also on CUDA for consistency)
    codec_ids_tensor = torch.as_tensor(codec_ids).to(device)
    
    len_prompt = prompt_ids.shape[-1]
    block_list = LogitsProcessorList([
        BlockTokenRangeProcessor(0, 46358),
        BlockTokenRangeProcessor(53526, tokenizer.vocab_size)
    ])
    
    # Teacher forcing loop - for each frame, add cb0 and generate 7 more codebooks
    num_frames = codec_ids_tensor.shape[1]
    for frame_idx in range(num_frames):
        cb0 = codec_ids_tensor[:, frame_idx:frame_idx+1]
        prompt_ids = torch.cat([prompt_ids, cb0], dim=1)
        
        with torch.no_grad():
            stage2_output = model_s2.generate(
                input_ids=prompt_ids,
                min_new_tokens=7,
                max_new_tokens=7,
                eos_token_id=tokenizer.eoa,
                pad_token_id=tokenizer.eoa,
                logits_processor=block_list,
            )
        
        # Keep prompt_ids on CUDA - mmgp expects CUDA inputs
        prompt_ids = stage2_output
        
        # Periodic cache cleanup every 50 frames to prevent fragmentation
        if frame_idx % 50 == 0 and frame_idx > 0:
            torch.cuda.empty_cache()
    
    # Final cleanup
    torch.cuda.empty_cache()
    
    # Extract only the generated tokens (after the initial prompt)
    # Move to CPU before converting to numpy
    output = prompt_ids[0].cpu().numpy()[len_prompt:]
    
    return output


def create_interface():
    """Create Gradio interface"""
    with gr.Blocks(title="YuE GPU-Poor") as interface:
        gr.Markdown("# YuE GPU-Poor - RTX 3060 12GB Optimized")
        gr.Markdown("Generate songs from lyrics using mmgp Profile 3 (12GB VRAM + 8-bit quantization)")
        
        with gr.Row():
            with gr.Column():
                genres_input = gr.Textbox(
                    label="Genre Tags (5 descriptors)",
                    placeholder="inspiring female uplifting pop airy vocal electronic bright vocal",
                    lines=2,
                    value="inspiring female uplifting pop airy vocal electronic bright vocal"
                )
                lyrics_input = gr.Textbox(
                    label="Lyrics (use [verse], [chorus], [outro] tags)",
                    placeholder="[verse]\nYour lyrics here\n\n[chorus]\nCatchy hook",
                    lines=10,
                    value="[verse]\nUnder the neon lights we dance\nLost in the rhythm of romance\n\n[chorus]\nWe are electric tonight\nShining so bright"
                )
                num_segments = gr.Slider(1, 5, value=2, step=1, label="Number of Segments")
                seed_input = gr.Number(value=42, label="Random Seed", precision=0)
                generate_btn = gr.Button("🎵 Generate Song", variant="primary")
            
            with gr.Column():
                status_output = gr.Textbox(label="Status", lines=2)
                mix_audio = gr.Audio(label="Mixed Output", type="filepath")
                vocal_audio = gr.Audio(label="Vocal Track", type="filepath")
        
        gr.Markdown("**Generation time:** ~4-6 minutes per 30-second segment on RTX 3060")
        
        generate_btn.click(
            fn=generate_song,
            inputs=[genres_input, lyrics_input, num_segments, seed_input],
            outputs=[status_output, mix_audio, vocal_audio]
        )
    
    return interface


if __name__ == "__main__":
    profile = int(os.getenv("YUE_PROFILE", "3"))
    initialize_models(profile=profile)
    
    interface = create_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
