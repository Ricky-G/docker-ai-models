# 🎵 YuE Quick Reference Card

## 🚀 Quick Start Commands

### Build Image
```powershell
cd yue
docker build -f dockerfile -t yue:latest .
```

### Run Web Interface (Recommended)
```powershell
docker run -it --rm --name yue `
  --gpus all --shm-size=16g -p 7860:7860 `
  -v "C:\_Models\yue\cache:/app/cache" `
  -v "C:\_Models\yue\output:/app/output" `
  -e YUE_MODE=web yue:latest
```
**Access**: http://localhost:7860

### Run CLI Mode
```powershell
docker run -it --rm --name yue `
  --gpus all --shm-size=16g `
  -v "C:\_Models\yue\cache:/app/cache" `
  -v "C:\_Models\yue\output:/app/output" `
  -e YUE_MODE=cli yue:latest
```

## 🎛️ Key Environment Variables

| Variable | Default | Options | Description |
|----------|---------|---------|-------------|
| `YUE_MODE` | `web` | `web`, `cli` | Interface mode |
| `YUE_STAGE1_MODEL` | `m-a-p/YuE-s1-7B-anneal-en-cot` | HF model ID | Stage 1 model |
| `YUE_STAGE2_MODEL` | `m-a-p/YuE-s2-1B-general` | HF model ID | Stage 2 model |
| `YUE_RUN_N_SEGMENTS` | `2` | 1-10 | Number of segments |
| `YUE_MAX_NEW_TOKENS` | `3000` | 500-6000 | Token limit |
| `YUE_SEED` | `42` | Any integer | Reproducibility |

## 🎨 Web Interface Features

### Text Prompts
- **Genre**: Include 5 components: genre, instrument, mood, vocal gender, vocal timbre
- **Lyrics**: Structure with sections: `[verse]`, `[chorus]`, `[bridge]`

### Audio Prompts (ICL)
- **Single Track**: Upload one audio file for style reference
- **Dual Track**: Upload separate vocal + instrumental tracks (recommended)
- **Timing**: Set start/end times (default: 0-30 seconds)

### Generation Controls
- **Real-time Logs**: Monitor progress with live updates
- **File Management**: Browse and play generated files
- **Process Control**: Start/stop generation safely

## 📁 Directory Structure

```
C:\_Models\yue\
├── cache/           # Model weights & cache
└── output/          # Generated music files
```

## 🔧 Troubleshooting

### Common Issues
- **Port 7860 in use**: Change port with `-p 7861:7860`
- **GPU not detected**: Ensure NVIDIA Docker is installed
- **Out of memory**: Reduce `YUE_STAGE2_BATCH_SIZE` or `YUE_RUN_N_SEGMENTS`
- **Web UI not loading**: Wait 2-3 minutes for models to load

### Check Container Logs
```powershell
docker logs yue
```

### Stop Container
```powershell
docker stop yue
```

## 🎵 Example Generation

### Genre Description Example
```
inspiring female uplifting pop airy vocal electronic bright vocal
```

### Lyrics Example
```
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
```

## 📊 Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU VRAM | 16GB | 24GB+ |
| System RAM | 16GB | 32GB+ |
| Storage | 50GB | 100GB+ |
| Docker | Latest | Latest with GPU support |

## 🌐 Web Interface URLs

- **Main Interface**: http://localhost:7860
- **Alternative Port**: http://localhost:7861 (if changed)

## 📝 Notes

- First run downloads ~10GB of models
- Generation takes 5-15 minutes depending on settings
- Output files are saved as WAV format
- Web interface supports real-time monitoring
- CLI mode outputs directly to console

---

**Need Help?** Check the full documentation in `README.md` or the integration summary in `GRADIO_INTEGRATION_SUMMARY.md`
