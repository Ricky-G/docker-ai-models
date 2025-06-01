# 🐳 docker-ai-models

**Production-Ready Dockerfiles for AI Models – Zero Setup, Maximum Results**

Transform complex AI model deployments into simple Docker commands. This repository provides optimized, tested containers for cutting-edge AI models across music, video, image, and text generation. No dependency hell, no environment conflicts – just pull, run, and create.

---

## 📦 Repository Structure

Each model has its own folder containing:

- `dockerfile` / `dockerfile-gpu-poor` – Container build files optimized for different hardware configurations
- `startup.sh` / `startup-gpu-poor.sh` – Automated setup and launch scripts
- Model-specific configurations and optimizations

Current structure:

```bash
├── yue/                    # YuE: Full-song music generation
│   ├── dockerfile          # Standard GPU build (24GB+ VRAM)
│   ├── dockerfile-gpu-poor # Lightweight build (8GB+ VRAM)
│   ├── startup.sh          # Standard launch script
│   └── startup-gpu-poor.sh # GPU-constrained launch script
├── wan/                    # Wan2GP: Video-to-audio generation
│   ├── dockerfile-gpu-poor # GPU-efficient build
│   └── startup-gpu-poor.sh # Launch script
```

---

## ✨ Featured AI Models

| Model | Type | Hardware | Interface | Documentation |
|-------|------|----------|-----------|---------------|
| 🎵 **YuE** | Lyrics-to-Song Generation | 24GB+ VRAM | Web UI + CLI | [Setup Guide](yue/README.md) |
| 🎶 **YuE-GP** | Music Generation (Optimized) | 8GB+ VRAM | Web UI | [Setup Guide](yue/README-GPU-POOR.md) |
| 🎬 **Wan2GP** | Video-to-Audio Synthesis | 8GB+ VRAM | Web UI | *Coming Soon* |

### 🚀 Quick Start

Choose your model and run a single command:

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/docker-ai-models.git
cd docker-ai-models

# Navigate to your chosen model
cd yue/  # or wan/

# Build and run (see individual README files for specific commands)
docker build -f dockerfile -t model:latest .
docker run --gpus all -p 7860:7860 model:latest
```

Open `http://localhost:7860` in your browser and start creating!

---

## 🎯 Why Choose This Repository?

- **🔧 Zero Configuration** – No dependency hunting, no environment conflicts
- **⚡ Hardware Optimized** – Multiple builds for different GPU capabilities
- **🌐 Web Interfaces** – Beautiful, intuitive web UIs for all models
- **📦 Production Ready** – Tested, stable containers ready for deployment
- **⏱️ Time Saving** – From hours of setup to minutes of runtime
- **🔄 Reproducible** – Same results, every time, everywhere

---

## 🛠️ How It Works

Each model directory contains:
- **Optimized Dockerfiles** for different hardware configurations
- **Automated startup scripts** handling all dependencies
- **Comprehensive documentation** with step-by-step guides
- **Web interfaces** providing intuitive controls and real-time feedback

---

## 🤝 Contributing

Have an AI model you'd like to containerize? We'd love your contribution!

1. Fork this repository
2. Create a new directory: `your-model-name/`
3. Add `dockerfile`, `startup.sh`, and `README.md`
4. Submit a pull request

---

## 📝 License

MIT License – free to use, modify, and distribute.

⭐ **Star this repo** if it saves you time!