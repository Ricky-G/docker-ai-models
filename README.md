# 🐳 docker-ai-models

**Ready-to-Run Dockerized AI Models – Zero Setup, Maximum Results**

Transform complex AI model deployments into simple Docker commands. This repository provides optimized, tested containers for cutting-edge AI models across music, comics, video, and text generation. No dependency hell, no environment conflicts – just build, run, and create.

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
│   ├── gradio_interface.py # Web interface
│   ├── startup.sh          # Standard launch script
│   └── startup-gpu-poor.sh # GPU-constrained launch script
├── seed-story/             # SEED-Story: Comic story generation
│   ├── dockerfile          # GPU build (8GB+ VRAM)
│   ├── minimal_gradio.py   # Main web interface
│   ├── simple_comic_generator.py # Fallback generator
│   ├── model_downloader.py # Model management
│   ├── startup.sh          # Launch script
│   └── requirements.txt    # Python dependencies
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
| 🎬 **SEED-Story** | Comic Story Generation | 8GB+ VRAM | Web UI | [Setup Guide](seed-story/README.md) |
| �️ **Wan2GP** | Video-to-Audio Synthesis | 8GB+ VRAM | Web UI | *Coming Soon* |

---

## 🔮 Coming Soon

Actively working on containerizing these cutting-edge AI models:

| Model | Type | Description | Repository |
|-------|------|-------------|------------|
| 📸 **PhotoMaker** | Image Generation | Customizing realistic human photos via stochastic identity mixing | [TencentARC/PhotoMaker](https://github.com/TencentARC/PhotoMaker) |
| 🗣️ **Fantasy Talking** | Video Generation | High-quality talking face generation with identity preservation | [Fantasy-AMAP/fantasy-talking](https://github.com/Fantasy-AMAP/fantasy-talking) |

*Watch this space – these containerized models will be available soon with the same zero-setup experience!*

### 🚀 Quick Start

Choose your model and run a single command:

```bash
# Clone the repository
git clone https://github.com/Ricky-G/docker-ai-models.git
cd docker-ai-models

# Navigate to your chosen model
cd seed-story/  # or yue/ or wan/

# Build and run (see individual README files for specific commands)
docker build -t seed-story .
docker run --gpus all -p 7860:7860 -v "C:\_Models\seed-story:/app/models" seed-story
```

Open `http://localhost:7860` in your browser and start creating!

---

## 🎯 Why Choose This Repository?

- **🔧 Zero Configuration** – No dependency hunting, no environment conflicts
- **⚡ Hardware Optimized** – Multiple builds for different GPU capabilities
- **🌐 Web Interfaces** – Beautiful, intuitive web UIs for all models
- **📦 Ready to Run** – Tested, stable containers that work out of the box
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