# docker-ai-models

Docker containers for AI models with GPU support.

This repository provides pre-configured Docker containers for various AI models. Each model includes:
- Complete environment setup with all dependencies
- GPU acceleration support via CUDA
- Web-based interface for interaction
- Persistent model storage to avoid re-downloading

The containers handle dependency management, CUDA configuration, and model downloads automatically. Models are stored in mounted volumes for reuse across container rebuilds.

**Target hardware**: Consumer GPUs (RTX series). All containers tested on RTX 3060 12GB VRAM.

**Memory optimization**: Uses [mmgp](https://pypi.org/project/mmgp/) (Memory Management for GPU Poor) to run large models on consumer hardware. mmgp enables models that normally require 24GB+ VRAM to run on 12GB cards through:
- 8-bit quantization of large model components
- Dynamic model offloading between VRAM and system RAM
- Selective layer loading (loads only active layers to VRAM)
- Reserved RAM pinning for fast transfers

Example: FLUX.1-schnell (22.7GB transformer + 8.8GB text encoder) runs on 12GB VRAM via quantization and partial pinning to system RAM.

**Use case**: Run inference on large AI models locally using retail GPUs without requiring datacenter hardware.

---

## Repository Structure

Each model directory contains:

- `dockerfile` - Container build configuration
- `startup.sh` - Container initialization script
- `gradio_interface.py` - Web UI implementation
- `requirements.txt` - Python dependencies

Directory layout:

```bash
├── yue/                    # YuE: Lyrics-to-song music generation
│   ├── dockerfile          # CUDA 12.4 runtime (12GB VRAM)
│   ├── gradio_interface.py # Web interface + generation pipeline
│   ├── startup.sh          # Container entrypoint
│   └── requirements.txt    # Python dependencies
├── seed-story/             # SEED-Story: Comic story generation
│   ├── dockerfile          # GPU build (8GB+ VRAM)
│   ├── minimal_gradio.py   # Main web interface
│   ├── simple_comic_generator.py # Fallback generator
│   ├── model_downloader.py # Model management
│   ├── startup.sh          # Launch script
│   └── requirements.txt    # Python dependencies
├── omnicontrol/            # OmniControlGP: Subject-driven image generation
│   ├── dockerfile          # GPU build (12GB+ VRAM)
│   ├── gradio_interface.py # Web interface
│   ├── startup.sh          # Launch script
│   ├── src/flux/           # FLUX model source code
│   └── requirements.txt    # Python dependencies
├── wan/                    # Wan2GP: Video-to-audio generation
│   ├── dockerfile-gpu-poor # GPU-efficient build
│   └── startup-gpu-poor.sh # Launch script
```

---

## Available Models

| Model | Type | Hardware | Interface | Documentation |
|-------|------|----------|-----------|---------------|
| 🎵 **YuE** | Lyrics-to-Song Generation | 12GB VRAM | Web UI | [Setup Guide](yue/README.md) |
| 🎬 **SEED-Story** | Comic Story Generation | 8GB+ VRAM | Web UI | [Setup Guide](seed-story/README.md) |
| 🎨 **OmniControl** | Subject-Driven Image Gen | 12GB+ VRAM | Web UI | [Setup Guide](omnicontrol/README.md) |
| 🎵️ **Wan2GP** | Video-to-Audio Synthesis | 8GB+ VRAM | Web UI | *Coming Soon* |

---

## Planned Models

| Model | Type | Description | Repository |
|-------|------|-------------|------------|
| 📸 **PhotoMaker** | Image Generation | Customizing realistic human photos via stochastic identity mixing | [TencentARC/PhotoMaker](https://github.com/TencentARC/PhotoMaker) |
| 🗣️ **Fantasy Talking** | Video Generation | High-quality talking face generation with identity preservation | [Fantasy-AMAP/fantasy-talking](https://github.com/Fantasy-AMAP/fantasy-talking) |

---

## Quick Start

```bash
# Clone the repository
git clone https://github.com/Ricky-G/docker-ai-models.git
cd docker-ai-models

# Navigate to your chosen model
cd omnicontrol/  # or yue/ or seed-story/ or wan/

# Build and run (see individual README files for specific commands)
docker build -t omnicontrol .
docker run -d --gpus all -p 7860:7860 \
  -v D:\_Models\omnicontrol:/app/models \
  -e HF_TOKEN=your_token_here \
  omnicontrol
```

Web interface available at `http://localhost:7860`

---

## Requirements

- Docker with NVIDIA GPU support
- NVIDIA Container Toolkit
- GPU with sufficient VRAM (see model-specific requirements)

---

## Contributing

To add a new model:

1. Fork the repository
2. Create a new directory for your model
3. Add dockerfile, startup script, and README
4. Submit a pull request

---

## License

MIT License