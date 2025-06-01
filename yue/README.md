# 🎵 YuE Official Implementation - Complete Music Generation

**High-Quality Lyrics-to-Song Generation with Official YuE Models**

This is the official YuE implementation with an enhanced Gradio web interface, supporting both web UI and CLI modes. Designed for high-end GPUs (24GB+ VRAM) to generate professional-quality full-length songs.

---

## ⚡ Quick Start (5 Minutes)

### 1. Prerequisites Check

**Required:**
- **High-end NVIDIA GPU** with 24GB+ VRAM (RTX 4090, A6000, H100, etc.)
- **Docker Desktop** with GPU support
- **32GB+ System RAM** recommended
- **Windows 10/11** with WSL2 or **Linux**

**Quick Check Commands:**
```powershell
# Check GPU (should show 24GB+ VRAM)
nvidia-smi

# Check Docker
docker --version

# Test GPU in Docker
docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu22.04 nvidia-smi
```

### 2. Clone and Build (3 Minutes)

```powershell
# Clone the repository
git clone https://github.com/YOUR_USERNAME/docker-ai-models.git
cd docker-ai-models/yue

# Build the official YuE image (this may take 15-20 minutes)
docker build -f dockerfile -t yue:latest .
```

### 3. Create Storage Directories

```powershell
# Create directories for models and outputs
mkdir C:\_Models\yue\cache
mkdir C:\_Models\yue\output
```

### 4. Run YuE Web Interface (Default)

```powershell
docker run -it --rm --name yue `
  --gpus all `
  --shm-size=16g `
  -p 7860:7860 `
  -v "C:\_Models\yue\cache:/app/cache" `
  -v "C:\_Models\yue\output:/app/output" `
  -e YUE_MODE=web `
  yue:latest
```

### 5. Access Web Interface

Open your browser to: **http://localhost:7860**

🎉 **Ready to create professional-quality music!**

---

## 🌐 Web Interface Features

### 🎯 Model Configuration
- **Stage 1 Model**: Text-to-semantic tokens (default: YuE-s1-7B-anneal-en-cot)
- **Stage 2 Model**: Semantic-to-audio tokens (default: YuE-s2-1B-general)
- **Real-time Model Selection**: Switch between different model variants

### 🎨 Text Prompts
- **Genre Description**: 5-component format (genre, instrument, mood, vocal gender, vocal timbre)
- **Structured Lyrics**: Support for `[verse]`, `[chorus]`, `[bridge]`, `[outro]` sections
- **Multi-language Support**: English, Chinese, Japanese, Korean

### 🎧 Audio Prompts (In-Context Learning)
- **Single-Track ICL**: Upload reference audio for style matching
- **Dual-Track ICL**: Separate vocal and instrumental tracks (recommended)
- **Timing Control**: Precise start/end time selection
- **High-Quality Processing**: 24kHz audio support

### ⚙️ Advanced Controls
- **Token Limit**: 500-6000 tokens (affects song length)
- **Batch Processing**: Multiple segments generation
- **Repetition Penalty**: Fine-tune lyrical repetition
- **Seed Control**: Reproducible generation
- **Real-time Monitoring**: Live progress tracking and logs

### 📁 File Management
- **Auto-refresh**: Automatically detect new generated files
- **Built-in Player**: Play generated music directly in browser
- **Download Support**: Direct download of generated audio
- **History Tracking**: View all previous generations

---

## 💻 CLI Mode (Advanced Users)

For automation, scripting, or batch processing:

```powershell
docker run -it --rm --name yue `
  --gpus all `
  --shm-size=16g `
  -v "C:\_Models\yue\cache:/app/cache" `
  -v "C:\_Models\yue\output:/app/output" `
  -e YUE_MODE=cli `
  -e YUE_STAGE1_MODEL=m-a-p/YuE-s1-7B-anneal-en-cot `
  -e YUE_STAGE2_MODEL=m-a-p/YuE-s2-1B-general `
  -e YUE_RUN_N_SEGMENTS=2 `
  yue:latest
```

**CLI Features:**
- Direct file output to mounted directories
- Scriptable and automatable
- Environment variable configuration
- Official `infer.py` interface

---

## 🎛️ Configuration Options

### Core Environment Variables

| Variable | Default | Options | Description |
|----------|---------|---------|-------------|
| `YUE_MODE` | `web` | `web`, `cli` | Interface mode |
| `YUE_STAGE1_MODEL` | `m-a-p/YuE-s1-7B-anneal-en-cot` | HuggingFace model ID | Stage 1 model path |
| `YUE_STAGE2_MODEL` | `m-a-p/YuE-s2-1B-general` | HuggingFace model ID | Stage 2 model path |
| `YUE_MAX_NEW_TOKENS` | `3000` | 500-6000 | Maximum tokens to generate |
| `YUE_RUN_N_SEGMENTS` | `2` | 1-10 | Number of song segments |
| `YUE_STAGE2_BATCH_SIZE` | `4` | 1-16 | Batch size for Stage 2 |
| `YUE_REPETITION_PENALTY` | `1.1` | 1.0-2.0 | Repetition penalty factor |
| `YUE_SEED` | `42` | Any integer | Random seed for reproducibility |

### Audio Prompt (ICL) Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `YUE_USE_AUDIO_PROMPT` | `0` | Enable single-track ICL (0/1) |
| `YUE_USE_DUAL_TRACKS_PROMPT` | `0` | Enable dual-track ICL (0/1) |
| `YUE_AUDIO_PROMPT_PATH` | - | Path to single audio prompt |
| `YUE_VOCAL_TRACK_PROMPT_PATH` | - | Path to vocal track |
| `YUE_INSTRUMENTAL_TRACK_PROMPT_PATH` | - | Path to instrumental track |
| `YUE_PROMPT_START_TIME` | `0.0` | Prompt start time (seconds) |
| `YUE_PROMPT_END_TIME` | `30.0` | Prompt end time (seconds) |

### Performance Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `YUE_KEEP_INTERMEDIATE` | `1` | Keep intermediate generation files |
| `YUE_DISABLE_OFFLOAD_MODEL` | `0` | Disable model offloading |
| `YUE_CUDA_IDX` | `0` | CUDA device index |
| `YUE_AUTO_UPDATE` | `0` | Auto-update repository on startup |

---

## 🎵 Music Generation Guide

### Genre Description Best Practices

**5-Component Format:**
1. **Genre**: pop, rock, jazz, electronic, classical, etc.
2. **Instruments**: guitar, piano, synthesizer, drums, etc.
3. **Mood**: uplifting, melancholic, energetic, calm, etc.
4. **Vocal Gender**: male, female, mixed, instrumental
5. **Vocal Timbre**: bright, airy, deep, soft, powerful, etc.

**Example:**
```
inspiring female uplifting pop airy vocal electronic bright synthesizer
```

### Lyrics Structure Guide

**Professional Song Structure:**
```
[intro]
(Optional instrumental opening)

[verse]
Main story content here
Each line should flow naturally
Building up to the chorus
Setting the emotional tone

[chorus]
Most memorable part of the song
Catchy and repeatable
Core message or hook
Emotional peak of the song

[verse]
Continue the story
Add new information
Build on verse 1
Lead to chorus again

[chorus]
Repeat for familiarity
Maybe slight variation
Emotional reinforcement
Audience participation point

[bridge]
Different melody/rhythm
New perspective on theme
Contrast to verse/chorus
Builds to final chorus

[chorus]
Final emotional statement
Maybe key change
Extended or modified
Strong ending feeling

[outro]
(Optional closing section)
```

### In-Context Learning (ICL) Tips

**Single-Track ICL:**
- Upload a high-quality reference song (MP3/WAV)
- Use 15-30 second segments
- Choose sections that represent the desired style
- Best for overall style matching

**Dual-Track ICL (Recommended):**
- Upload separate vocal and instrumental tracks
- Provides more precise style control
- Better quality for complex arrangements
- Use professional stems when available

**Audio Quality Guidelines:**
- **Sample Rate**: 44.1kHz or higher
- **Bit Depth**: 16-bit minimum, 24-bit preferred
- **Format**: WAV preferred, high-quality MP3 acceptable
- **Length**: 15-60 seconds optimal

---

## 🚨 Troubleshooting

### Common Issues & Solutions

**❌ "CUDA out of memory" Error**
```powershell
# Reduce batch size and segments
docker run -it --rm --name yue `
  --gpus all --shm-size=16g -p 7860:7860 `
  -v "C:\_Models\yue\cache:/app/cache" `
  -v "C:\_Models\yue\output:/app/output" `
  -e YUE_MODE=web `
  -e YUE_STAGE2_BATCH_SIZE=2 `
  -e YUE_RUN_N_SEGMENTS=1 `
  yue:latest
```

**❌ "Models not downloading"**
- Check internet connection
- Ensure HuggingFace access (some models may require login)
- Verify sufficient disk space (>50GB)
- Check container logs: `docker logs yue`

**❌ "Web interface not accessible"**
- Wait 3-5 minutes for complete initialization
- Check port conflicts: `netstat -an | findstr 7860`
- Try alternative port: `-p 7861:7860`
- Verify firewall settings

**❌ "Generation taking too long"**
- First generation always takes longer (model loading)
- Reduce `YUE_MAX_NEW_TOKENS` for shorter songs
- Reduce `YUE_RUN_N_SEGMENTS` for faster generation
- Monitor GPU usage: `nvidia-smi`

### Advanced Debugging

**Check Container Status:**
```powershell
# View running containers
docker ps

# Check detailed logs
docker logs yue -f

# Access container shell
docker exec -it yue bash

# Monitor GPU usage
nvidia-smi -l 1
```

**Performance Monitoring:**
```powershell
# System resources
docker stats yue

# GPU memory usage
nvidia-smi --query-gpu=memory.used,memory.total --format=csv

# Disk usage
docker exec yue df -h
```

---

## 🔄 Updates and Maintenance

### Regular Updates
```powershell
# Update repository
cd docker-ai-models/yue
git pull

# Rebuild with latest changes
docker build -f dockerfile -t yue:latest .

# Update models (optional)
docker run -it --rm --name yue-update `
  --gpus all --shm-size=16g `
  -v "C:\_Models\yue\cache:/app/cache" `
  -e YUE_AUTO_UPDATE=1 `
  -e YUE_MODE=cli `
  yue:latest
```

### Disk Space Management
```powershell
# Check disk usage
docker system df

# Clean up containers
docker container prune

# Clean up images
docker image prune

# Clean up build cache
docker builder prune

# Full cleanup (careful!)
docker system prune --volumes
```

### Model Management
```powershell
# Check model cache size
du -sh C:\_Models\yue\cache

# Backup important models
xcopy "C:\_Models\yue\cache" "C:\_Backup\yue\cache" /E /I

# Clear model cache (will re-download)
rmdir /S "C:\_Models\yue\cache"
mkdir "C:\_Models\yue\cache"
```

---

## 📊 Performance Guidelines

### Hardware Recommendations

| Component | Minimum | Recommended | Optimal |
|-----------|---------|-------------|---------|
| **GPU VRAM** | 20GB | 24GB | 48GB+ |
| **GPU Model** | RTX 4090 | RTX 4090 | H100/A100 |
| **System RAM** | 32GB | 64GB | 128GB+ |
| **Storage** | 100GB SSD | 500GB NVMe | 1TB+ NVMe |
| **CPU** | 8 cores | 16 cores | 32+ cores |

### Expected Performance

| GPU Model | VRAM | Generation Time | Quality Level |
|-----------|------|----------------|---------------|
| RTX 4090 | 24GB | 3-8 minutes | Excellent |
| RTX A6000 | 48GB | 2-5 minutes | Excellent |
| H100 | 80GB | 1-3 minutes | Maximum |
| A100 | 40GB | 2-6 minutes | Excellent |

### Optimization Tips

**For Maximum Quality:**
```bash
YUE_STAGE2_BATCH_SIZE=1
YUE_MAX_NEW_TOKENS=6000
YUE_RUN_N_SEGMENTS=4
YUE_DISABLE_OFFLOAD_MODEL=1
```

**For Speed:**
```bash
YUE_STAGE2_BATCH_SIZE=8
YUE_MAX_NEW_TOKENS=2000
YUE_RUN_N_SEGMENTS=2
YUE_DISABLE_OFFLOAD_MODEL=0
```

**For Balanced Performance:**
```bash
YUE_STAGE2_BATCH_SIZE=4
YUE_MAX_NEW_TOKENS=3000
YUE_RUN_N_SEGMENTS=2
YUE_DISABLE_OFFLOAD_MODEL=0
```

---

## 🆘 Support & Community

### Getting Help
- **GitHub Issues**: Technical problems and bug reports
- **Discussions**: Share your generated music and get feedback
- **Wiki**: Comprehensive guides and advanced tutorials
- **Discord**: Real-time community chat (link in repo)

### Contributing
- **Bug Reports**: Use the issue template
- **Feature Requests**: Describe your use case
- **Pull Requests**: Code contributions welcome
- **Documentation**: Help improve these guides

### Quick Reference Commands
```powershell
# Standard web interface
docker run -it --rm --name yue --gpus all --shm-size=16g -p 7860:7860 -v "C:\_Models\yue\cache:/app/cache" -v "C:\_Models\yue\output:/app/output" -e YUE_MODE=web yue:latest

# CLI mode
docker run -it --rm --name yue --gpus all --shm-size=16g -v "C:\_Models\yue\cache:/app/cache" -v "C:\_Models\yue\output:/app/output" -e YUE_MODE=cli yue:latest

# Check logs
docker logs yue -f

# Stop container
docker stop yue
```

---

## 📝 Technical Details

### Model Architecture
- **Stage 1**: Text-to-semantic token generation (7B parameters)
- **Stage 2**: Semantic-to-audio token generation (1B parameters)
- **Audio Codec**: High-quality audio reconstruction
- **Training Data**: Multilingual music dataset

### Output Specifications
- **Format**: WAV (uncompressed)
- **Sample Rate**: 24kHz
- **Bit Depth**: 16-bit
- **Channels**: Stereo
- **Duration**: Variable (based on token count)

### API Compatibility
- Full compatibility with official YuE repository
- Support for all official model variants
- Command-line interface identical to original
- Environment variable configuration

---

**🎵 Create Professional Music with YuE!**

This official implementation provides maximum quality and flexibility for serious music generation projects. Start with the web interface for ease of use, then explore CLI mode for advanced workflows.
