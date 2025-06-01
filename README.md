# 🐳 docker-ai-models

**Prebuilt Dockerfiles for AI Music Generation Models – Build and Run with Zero Setup**

This repository provides ready-to-use Dockerfiles and supporting scripts for AI music generation models. Whether you're experimenting with lyrics-to-song generation, testing different configurations, or deploying for inference, this repo helps you get started with minimal setup and no dependency headaches.

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

## 🔧 YuE Implementation Guide

This repository provides two different YuE implementations optimized for different hardware configurations:

### 🎯 **Official YuE (High-End GPUs)**

- **Files**: `dockerfile`, `startup.sh`
- **Interface**: **Web UI** (default) at http://localhost:7860 or **CLI mode**
- **Source**: Direct from [multimodal-art-projection/YuE](https://github.com/multimodal-art-projection/YuE)
- **Requirements**: 24GB+ VRAM (RTX 4090, A6000, H100, etc.)
- **Output**: Professional-quality full-length songs with advanced features
- **Best for**: Production use, professional music generation, maximum quality

📖 **[Complete Setup Guide →](yue/README.md)**

### 🎮 **YuE-GP (Consumer GPUs)**

- **Files**: `dockerfile-gpu-poor`, `startup-gpu-poor.sh`
- **Interface**: Gradio web UI at http://localhost:7860
- **Source**: Community project [deepbeepmeep/YuEGP](https://github.com/deepbeepmeep/YuEGP)
- **Requirements**: 8GB+ VRAM (RTX 3070, RTX 4060, RTX 4070, etc.)
- **Output**: High-quality music generation optimized for limited VRAM
- **Best for**: Experimentation, consumer hardware, quick setup

📖 **[Quick Start Guide →](yue/README-GPU-POOR.md)**

---

## 🚀 Getting Started

### Prerequisites

- **Docker** with GPU support (NVIDIA Container Toolkit)
- **NVIDIA GPU** with sufficient VRAM:
  - **High-end builds**: 24GB+ VRAM (RTX 4090, A6000, H100)
  - **Consumer builds**: 8GB+ VRAM (RTX 3070, RTX 4060, RTX 4070)
- **Windows PowerShell** or **Linux/WSL terminal**

### Quick Start

**🎯 Choose Your Setup Guide:**

- **High-End GPU (24GB+ VRAM)**: Follow the [Complete YuE Setup Guide](yue/README.md)
- **Consumer GPU (8GB+ VRAM)**: Follow the [Quick Start Guide](yue/README-GPU-POOR.md)

Both guides include detailed instructions for building, running, and using the web interface.

#### Alternative: Manual Setup Examples

If you prefer to see the commands directly, here are quick examples:

**For High-End GPUs (24GB+ VRAM) - Official YuE:**

```powershell
# Clone and build
git clone https://github.com/YOUR_USERNAME/docker-ai-models.git
cd docker-ai-models/yue
docker build -f dockerfile -t yue:latest .

# Run web interface (default)
docker run -it --rm --name yue --gpus all --shm-size=16g -p 7860:7860 -v "C:\_Models\yue\cache:/app/cache" -v "C:\_Models\yue\output:/app/output" -e YUE_MODE=web yue:latest
```

**For Consumer GPUs (8GB+ VRAM) - YuE-GP:**

```powershell
# Clone and build
git clone https://github.com/YOUR_USERNAME/docker-ai-models.git
cd docker-ai-models/yue
docker build -f dockerfile-gpu-poor -t yue-gpu-poor:latest .

# Run web interface
docker run -it --rm --name yue-gpu-poor --gpus all --shm-size=8g -p 7860:7860 -v "C:\_Models\yue:/workspace" yue-gpu-poor:latest
```

**Access the Interface:**
Open your browser to `http://localhost:7860` and start generating music!

---

## 🤖 Included Models

| Model Name    | Description                              | Documentation                                                | Interface Type           |
|---------------|------------------------------------------|--------------------------------------------------------------|--------------------------|
| **YuE**       | Official lyrics-to-song music generation | [Complete Setup Guide](yue/README.md) | **Web UI** + CLI modes  |
| **YuE-GP**    | GPU-optimized YuE for limited VRAM      | [Quick Start Guide](yue/README-GPU-POOR.md)   | Gradio Web UI (8GB+ VRAM)|
| **Wan2GP**    | Video-to-audio generation AI             | *Coming soon*                                                | Gradio Web UI            |

### Model Capabilities

**YuE Series:**

- Generate complete songs from lyrics input
- Support for multiple languages (English, Chinese, Japanese, Korean)
- Voice cloning and style transfer capabilities
- In-context learning with reference audio
- **Official YuE**: Professional **Gradio Web UI** (default) with optional CLI mode using the official `infer.py` script
- **YuE-GP**: Community Gradio web interface optimized for consumer GPUs

**📖 Detailed Setup Instructions:**
- **High-End GPU Setup**: [Complete YuE Guide](yue/README.md)
- **Consumer GPU Setup**: [Quick Start Guide](yue/README-GPU-POOR.md)

**Wan2GP:**

- Video-to-audio synthesis
- GPU memory optimized for consumer hardware
- Web-based interface for file processing

---

## 🧠 Why Use This Repository?

- **Zero Setup Hassle:** No more hunting down dependencies or fixing broken installs
- **Hardware Optimized:** Multiple build configurations for different GPU memory constraints  
- **Portable Environments:** Reproducible Docker containers that work everywhere
- **Production Ready:** Great for experiments, benchmarks, and production prototypes
- **Time Saving:** Save hours setting up each time you want to try a new AI music model
- **Web Interfaces:** All models include easy-to-use Gradio web interfaces
- **Detailed Guides:** Step-by-step instructions for both beginners and advanced users

---

## 🛠️ Advanced Configuration

For detailed configuration options, environment variables, and troubleshooting:

- **Official YuE**: See [Complete Setup Guide](yue/README.md)
- **YuE-GP**: See [Quick Start Guide](yue/README-GPU-POOR.md)

### Basic Environment Variables

**YuE Models:**

- `YUE_MODE`: Interface mode - `web` (default) or `cli`
- `YUE_STAGE1_MODEL`: Stage 1 model path
- `YUE_STAGE2_MODEL`: Stage 2 model path
- `YUE_RUN_N_SEGMENTS`: Number of song segments

**YuE-GP Models:**

- `YUEGP_PROFILE`: Performance profile (1=balanced, 2=speed, 3=quality)
- `YUEGP_CUDA_IDX`: CUDA device index

---

## 🤝 Contributing

Want to add a model? PRs are welcome!

1. Create a new folder under `<your-model-name>/`
2. Add a `dockerfile`, optional `dockerfile-gpu-*` variants, and a model-specific `README.md`
3. Follow the structure of existing models
4. Include clear setup instructions and examples

---

## 📚 Additional Resources

- **[YuE Quick Reference](YUE_QUICK_REFERENCE.md)** - Command reference and tips
- **GitHub Issues** - Report bugs or ask questions
- **GitHub Discussions** - Share your generated music and get feedback

---

## 📝 License

MIT License – free to use, modify, and distribute.

If you find this helpful, a ⭐️ or mention is always appreciated!