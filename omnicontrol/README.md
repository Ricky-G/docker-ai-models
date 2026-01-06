# OmniControlGP

Subject-driven image generation with FLUX.1-schnell + OminiControl LoRA.

## Quick Start

```powershell
# Build
docker build -t omnicontrol .

# Run (models auto-download to D:\_Models\omnicontrol on first run)
docker run -d --gpus all --name omnicontrol -p 7860:7860 `
  -v D:\_Models\omnicontrol:/app/models `
  -e HF_TOKEN=your_token_here `
  omnicontrol
```

Access at http://localhost:7860

## Requirements

- **VRAM**: 12GB minimum (Profile 3)
- **Storage**: 50GB for models (FLUX.1-schnell ~34GB + LoRA ~200MB)
- **First run**: 5-10 minutes to load models from disk to RAM/VRAM
- **Generation**: ~10 seconds after initial load

## Usage

1. Upload object image (center-cropped to 512x512)
2. Write prompt referring to object as "this item" or "it"
3. Click Generate

**Example prompt:**
```
"A film photography shot. This item is placed on a wooden desk 
in a cozy study room. Warm afternoon sunlight streams through the window."
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OMNI_PROFILE` | `3` | Memory profile (3 = 12GB VRAM) |
| `HF_TOKEN` | - | HuggingFace token for model downloads |
| `GRADIO_SERVER_PORT` | `7860` | Web UI port |

## Troubleshooting

**Out of Memory**: Restart container to clear VRAM
**Slow loading**: Models load from disk once, then stay in VRAM
**Models not found**: Check HF_TOKEN and internet connection

---

Memory optimization via [mmgp](https://pypi.org/project/mmgp/) (Memory Management for GPU Poor)
