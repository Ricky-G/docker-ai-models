# 🎵 YuE GPU-Poor Setup - Quick Start Guide

**GPU-Optimized YuE Music Generation for 8GB+ VRAM GPUs**

This guide will get you up and running with YuE music generation in under 10 minutes, even with limited GPU memory (8GB+ VRAM). Perfect for consumer GPUs like RTX 3070, RTX 4060, or similar.

---

## ⚡ Quick Start (5 Minutes)

### 1. Prerequisites Check

**Required:**
- **NVIDIA GPU** with 8GB+ VRAM (RTX 3070, RTX 4060, RTX 4070, etc.)
- **Docker Desktop** with GPU support
- **Windows 10/11** with WSL2 or **Linux**

**Quick Check Commands:**
```powershell
# Check GPU
nvidia-smi

# Check Docker
docker --version

# Test GPU in Docker
docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu22.04 nvidia-smi
```

### 2. Clone and Build (2 Minutes)

```powershell
# Clone the repository
git clone https://github.com/YOUR_USERNAME/docker-ai-models.git
cd docker-ai-models/yue

# Build the GPU-poor optimized image (this may take 10-15 minutes)
docker build -f dockerfile-gpu-poor -t yue-gpu-poor:latest .
```

### 3. Create Storage Directories

```powershell
# Create directories for models and outputs
mkdir C:\_Models\yue
```

### 4. Run YuE (1 Command)

```powershell
docker run -it --rm --name yue-gpu-poor `
  --gpus all `
  --shm-size=8g `
  -p 7860:7860 `
  -v "C:\_Models\yue:/workspace" `
  -e YUEGP_PROFILE=1 `
  yue-gpu-poor:latest
```

### 5. Access Web Interface

Open your browser to: **http://localhost:7860**

⏱️ **First Run**: Allow 3-5 minutes for the web interface to load (models are being downloaded and loaded)

🎉 **That's it!** You're ready to generate music!

---

## 🎛️ Web Interface Guide

### First Generation
1. **Keep default settings** (optimized for 8GB VRAM)
2. **Enter lyrics** in the text box (or use the example)
3. **Click "Generate"**
4. **Wait 5-10 minutes** for your first song
5. **Download** the generated audio file

### Example Lyrics Format
```
[verse]
Walking through the neon lights tonight
City sounds are calling out my name
Every step feels like a brand new start
Nothing ever gonna be the same

[chorus]  
We are the dreamers in the dark
Dancing to our beating hearts
Rising up above the noise
This is where we make our choice
```

### Performance Tips
- **First run**: Downloads ~3GB of models (be patient!)
- **Generation time**: 5-15 minutes depending on song length
- **Memory usage**: ~6-8GB VRAM during generation
- **Best quality**: Use shorter segments for better results

---

## 🔧 Configuration Options

### Environment Variables

| Variable | Default | Description | GPU-Poor Recommendation |
|----------|---------|-------------|-------------------------|
| `YUEGP_PROFILE` | `1` | Performance profile (1=balanced, 2=speed, 3=quality) | Keep at `1` |
| `YUEGP_CUDA_IDX` | `0` | GPU device index | `0` for single GPU |
| `YUEGP_AUTO_UPDATE` | `0` | Auto-update on startup | `0` for faster startup |

### Custom Run Command
```powershell
# For different performance profiles
docker run -it --rm --name yue-gpu-poor `
  --gpus all --shm-size=8g -p 7860:7860 `
  -v "C:\_Models\yue:/workspace" `
  -e YUEGP_PROFILE=2 `  # 2=speed, 3=quality
  yue-gpu-poor:latest
```

---

## 🚨 Troubleshooting

### Common Issues & Fixes

**❌ "CUDA out of memory" Error**
```powershell
# Reduce memory usage
docker run -it --rm --name yue-gpu-poor `
  --gpus all --shm-size=4g -p 7860:7860 `
  -v "C:\_Models\yue:/workspace" `
  -e YUEGP_PROFILE=1 `
  yue-gpu-poor:latest
```

**❌ "Port 7860 is already in use"**
```powershell
# Use different port
docker run -it --rm --name yue-gpu-poor `
  --gpus all --shm-size=8g -p 7861:7860 `
  -v "C:\_Models\yue:/workspace" `
  yue-gpu-poor:latest
```
Access at: http://localhost:7861

**❌ "Cannot connect to Docker daemon"**
- Start Docker Desktop
- Enable WSL2 integration in Docker Desktop settings

**❌ "No such file or directory" (Linux)**
```bash
# Linux users: adjust the volume mount
docker run -it --rm --name yue-gpu-poor \
  --gpus all --shm-size=8g -p 7860:7860 \
  -v "$HOME/Models/yue:/workspace" \
  yue-gpu-poor:latest
```

**❌ Web interface not loading**
- Wait 3-5 minutes for models to download/load
- Check container logs: `docker logs yue-gpu-poor`

### Check Container Status
```powershell
# View running containers
docker ps

# Check logs
docker logs yue-gpu-poor

# Stop container
docker stop yue-gpu-poor
```

---

## 🎵 Music Generation Tips

### Best Practices
- **Lyrics length**: 4-8 lines per section for 8GB VRAM
- **Song structure**: Use `[verse]`, `[chorus]`, `[bridge]` sections
- **Generation time**: First generation takes longer (model loading)
- **Quality vs Speed**: Profile 1 offers best balance for limited VRAM

### Example Prompts That Work Well
```
[verse]
Late night driving through the empty streets
Headlights cutting through the misty air
Radio playing our favorite beats
Nothing else matters when you're there

[chorus]
This is our moment, this is our time
Everything falling into line
We got the whole world in our hands
Living life like we planned
```

### Advanced Features
- **Batch generation**: Generate multiple short songs
- **Style consistency**: Use similar lyrics structure for consistent results
- **Memory optimization**: Close other GPU applications during generation

---

## 📁 File Management

### Directory Structure
```
C:\_Models\yue\           # Your mounted directory
├── models/               # Downloaded model files
├── outputs/              # Generated music files
└── cache/               # Temporary files
```

### Generated Files
- **Location**: `C:\_Models\yue\outputs\`
- **Format**: WAV audio files
- **Naming**: Timestamped (e.g., `output_20250601_143022.wav`)
- **Size**: ~10-50MB per song

---

## 🔄 Updates and Maintenance

### Update the Container
```powershell
# Pull latest changes
cd docker-ai-models/yue
git pull

# Rebuild image
docker build -f dockerfile-gpu-poor -t yue-gpu-poor:latest .
```

### Clean Up Disk Space
```powershell
# Remove old containers
docker container prune

# Remove unused images
docker image prune

# Remove everything (careful!)
docker system prune
```

---

## 💡 Performance Comparison

| GPU Model | VRAM | Recommended Profile | Est. Generation Time |
|-----------|------|-------------------|---------------------|
| RTX 3060 | 12GB | Profile 1 | 8-12 minutes |
| RTX 3070 | 8GB | Profile 1-2 | 6-10 minutes |
| RTX 4060 | 8GB | Profile 1-2 | 5-8 minutes |
| RTX 4070 | 12GB | Profile 1-3 | 4-7 minutes |

---

## 🆘 Need Help?

### Community Support
- **GitHub Issues**: Report bugs or ask questions
- **Discussions**: Share generated music and tips
- **Wiki**: Additional guides and examples

### Quick Commands Reference
```powershell
# Start YuE
docker run -it --rm --name yue-gpu-poor --gpus all --shm-size=8g -p 7860:7860 -v "C:\_Models\yue:/workspace" yue-gpu-poor:latest

# Stop YuE
docker stop yue-gpu-poor

# Check logs
docker logs yue-gpu-poor

# Update
git pull && docker build -f dockerfile-gpu-poor -t yue-gpu-poor:latest .
```

### System Requirements Summary
- **Minimum**: NVIDIA GPU 8GB VRAM, 16GB RAM, 20GB storage
- **Recommended**: NVIDIA GPU 12GB+ VRAM, 32GB RAM, 50GB storage
- **OS**: Windows 10/11 with WSL2, Ubuntu 20.04+, or similar Linux

---

**🎵 Happy Music Making!** 

Your GPU-poor setup is optimized for quality music generation even with limited VRAM. Start with the default settings and experiment from there!
