# YuE GPU-Poor - Dockerized

🎵 **Lyrics-to-Song Generation** optimized for RTX 3060 12GB VRAM using `mmgp` Profile 3 with 8-bit quantization.

---

## ✅ Features

- **Two-Stage Generation**: 7B Stage 1 (lyrics→semantic) + 1B Stage 2 (semantic→audio)
- **Aggressive VRAM Management**: mmgp offloading, chunked audio decode, smart cleanup
- **High-Quality Output**: 44.1kHz stereo audio with separate vocal/instrumental tracks
- **Gradio Web UI**: Simple interface at http://localhost:7860

---

## 📋 Requirements

| Component | Requirement |
|-----------|-------------|
| **GPU** | RTX 3060 12GB VRAM (or better) |
| **RAM** | 64GB system memory |
| **Storage** | ~50GB free for models |
| **Docker** | Desktop with WSL2 + NVIDIA GPU support |

---

## 🚀 Quick Start

### 1. Build
```powershell
cd C:\_Github\docker-ai-models\yue
docker build -t yue:latest .
```

### 2. Run (First Time)
```powershell
# Create persistent storage
mkdir D:\_Models\yue

# Run container
docker run -d --name yue `
  --gpus all `
  --shm-size=16g `
  -p 7860:7860 `
  -e HF_TOKEN=your_token_here `
  -v "D:\_Models\yue:/app/models" `
  yue:latest
```

### 3. Monitor Startup
```powershell
docker logs yue -f
```

**First startup (~15-20 min):**
- Installs PyTorch 2.5.1 + CUDA 12.4
- Downloads mmgp, transformers, gradio
- Downloads xcodec codec model
- YuE models download on first generation (~8GB)

**Subsequent runs (~30 sec):**
- Uses cached pip packages
- Models already downloaded
- Ready when: `Running on http://0.0.0.0:7860`

### 4. Access
Open http://localhost:7860

---

## 🎤 Usage

### Genre Tags
Enter 5+ descriptors separated by spaces:
```
inspiring female uplifting pop airy vocal electronic bright
```

### Lyrics Format
Use section tags:
```
[verse]
Under the neon lights we dance
Lost in the rhythm of romance

[chorus]  
We are electric tonight
Shining so bright

[outro]
The night fades away...
```

### Generation Time
- **~4-6 min per 30-second segment** on RTX 3060
- 2 segments = ~10-12 min total

---

## 📁 File Structure

```
yue/
├── dockerfile           # CUDA 12.4 runtime container
├── gradio_interface.py  # Web UI + generation pipeline  
├── requirements.txt     # Python dependencies
├── startup.sh          # Container entrypoint
└── README.md           # This file
```

---

## ⚙️ Configuration

### Environment Variables
| Variable | Default | Description |
|----------|---------|-------------|
| `HF_TOKEN` | (required) | HuggingFace token for model downloads |
| `YUE_PROFILE` | `3` | mmgp optimization profile |

**Get HF Token:** https://huggingface.co/settings/tokens (read access)

### mmgp Profiles
| Profile | VRAM | Description |
|---------|------|-------------|
| 1 | 16GB | No optimization |
| 3 | 12GB | 8-bit quantization (default) |
| 4 | 10GB | Sequential offload |
| 5 | 8GB | Aggressive offload |

---

## 🔧 Technical Details

### Models (Auto-Downloaded)
| Model | Size | Purpose |
|-------|------|---------|
| `m-a-p/YuE-s1-7B-anneal-en-cot` | 7B | Lyrics → semantic tokens |
| `m-a-p/YuE-s2-1B-general` | 1B | Semantic → audio tokens |
| `m-a-p/xcodec_mini_infer` | ~200MB | Audio codec |

### Optimizations for 12GB VRAM
- **mmgp Profile 3**: 8-bit quantization for both stage models
- **Stage Offloading**: Stage 1 offloaded before Stage 2 starts
- **Codec Offloading**: Codec offloaded during Stage 2
- **Chunked Decode**: 500-frame chunks for codec/vocoder decoding
- **mmgp unload_all()**: Force-releases GPU memory between stages

### Output Specs
| Property | Value |
|----------|-------|
| Format | WAV |
| Sample Rate | 44.1kHz |
| Bit Depth | 16-bit |
| Tracks | Vocal, Instrumental, Mix |

### Storage Paths (Container)
| Path | Purpose |
|------|---------|
| `/app/models` | HuggingFace cache, PyTorch cache |
| `/app/models/pip` | Cached pip packages |
| `/app/output` | Generated audio files |

---

## 🛠️ Troubleshooting

### Out of Memory
```powershell
# Try Profile 4 (10GB mode)
docker run -d --name yue --gpus all --shm-size=16g -p 7860:7860 `
  -e HF_TOKEN=your_token -e YUE_PROFILE=4 `
  -v "D:\_Models\yue:/app/models" yue:latest
```

### GPU Not Detected
```powershell
# Verify Docker GPU support
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

### Slow First Generation
- YuE models (~8GB) download from HuggingFace on first use
- Check `D:\_Models\yue` for cached files
- Subsequent generations are faster

### Container Management
```powershell
# View logs
docker logs yue -f

# Restart
docker restart yue

# Stop and remove
docker stop yue && docker rm yue

# Rebuild after code changes
docker stop yue; docker rm yue; docker build -t yue:latest .; docker run -d --name yue --gpus all --shm-size=16g -p 7860:7860 -e HF_TOKEN=your_token -v "D:\_Models\yue:/app/models" yue:latest
```

---

## 📊 Expected VRAM Usage

| Stage | VRAM (Peak) |
|-------|-------------|
| Stage 1 (7B) | ~11GB |
| After S1 offload | ~7.4GB |
| After codec offload | ~6.7GB |
| Stage 2 (1B) | ~10.4GB |
| After mmgp unload | ~0.5GB |
| Codec decode | ~0.7GB |
| Vocoder upsample | ~1GB |

---

**🎵 Create professional music from lyrics with YuE!**
