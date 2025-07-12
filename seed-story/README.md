# 🎬 SEED-Story Docker Setup

**Multimodal Long Story Generation with Large Language Model**

> Generate rich, coherent multimodal stories with consistent characters and style. SEED-Story creates visual narratives that span up to 25 sequences with both narrative text and generated images.

[![arXiv](https://img.shields.io/badge/arXiv-2407.08683-red.svg)](https://arxiv.org/abs/2407.08683)
[![Hugging Face](https://img.shields.io/badge/🤗-Models-yellow.svg)](https://huggingface.co/TencentARC/SEED-Story)
[![GitHub](https://img.shields.io/badge/GitHub-TencentARC/SEED--Story-blue.svg)](https://github.com/TencentARC/SEED-Story)

## 🌟 Features

- **🎨 Multimodal Generation**: Creates both narrative text and corresponding images
- **🔗 Character Consistency**: Maintains consistent characters and style throughout the story
- **📚 Long-form Stories**: Generates stories up to 25 sequences long
- **🌐 Web Interface**: Beautiful Gradio web UI for easy interaction
- **🐳 Docker Ready**: Complete containerized solution with GPU support
- **⚙️ Configurable**: Adjustable story length, generation parameters, and styles

## 🚀 Quick Start

### Prerequisites

- **GPU**: NVIDIA GPU with 16GB+ VRAM (24GB+ recommended)
- **Storage**: ~32GB for all models
- **Docker**: Docker with NVIDIA Container Runtime
- **Memory**: 16GB+ system RAM

### 1. Build the Container

```bash
# Clone the repository
git clone <repository-url>
cd docker-ai-models/seed-story

# Build the Docker image
docker build -t seed-story .
```

### 2. Run Web Interface

```bash
# Start web interface (default mode)
docker run --gpus all -p 7860:7860 -v $(pwd)/data:/app/data seed-story

# Access the interface
open http://localhost:7860
```

### 3. CLI Mode

```bash
# Interactive CLI mode
docker run --gpus all -e SEED_STORY_MODE=cli -it seed-story

# Run specific script
docker run --gpus all -e SEED_STORY_MODE=cli seed-story src/inference/gen_george.py
```

## 📋 Model Requirements

The following models will be downloaded automatically on first run:

| Model | Size | Description |
|-------|------|-------------|
| **Stable Diffusion XL** | ~7GB | Base image generation model |
| **Llama-2-7b-hf** | ~13GB | Language model for text generation |
| **Qwen-VL-Chat** | ~10GB | Vision-language model for understanding |
| **SEED-Story Checkpoints** | ~2GB | Pre-trained SEED-Story weights |

**Total Storage**: ~32GB + working space

### Manual Model Download

```bash
# Download models manually
docker run --gpus all -e SEED_STORY_MODE=cli seed-story python3 model_downloader.py

# List required models
docker run --gpus all -e SEED_STORY_MODE=cli seed-story python3 model_downloader.py --list
```

## 🎯 Usage Examples

### Web Interface

1. **Upload an Image**: Provide a starting image for your story
2. **Enter Opening Text**: Write the beginning of your story
3. **Configure Settings**: Adjust story length and generation parameters
4. **Generate**: Click "Generate Story" and watch your multimodal story unfold

### CLI Examples

```bash
# Generate story with custom parameters
docker run --gpus all -e SEED_STORY_MODE=cli seed-story \
  python3 src/inference/gen_george.py

# Visualize story with attention
docker run --gpus all -e SEED_STORY_MODE=cli seed-story \
  python3 src/inference/vis_george_sink.py
```

## ⚙️ Configuration

### Environment Variables

```bash
SEED_STORY_MODE=web          # 'web' or 'cli'
CUDA_VISIBLE_DEVICES=0       # GPU device selection
GRADIO_SERVER_NAME=0.0.0.0   # Web interface host
GRADIO_SERVER_PORT=7860      # Web interface port
```

### Generation Parameters

- **Story Length**: 3-25 sequences (default: 10)
- **Max Tokens**: 100-1000 per segment (default: 500)
- **Inference Steps**: 20-100 for image generation (default: 50)
- **Temperature**: 0.1-1.5 for text creativity (default: 0.7)
- **Guidance Scale**: 1.0-15.0 for image fidelity (default: 7.5)

## 📁 Data Persistence

```bash
# Mount data directory for persistence
docker run --gpus all -p 7860:7860 \
  -v $(pwd)/models:/app/pretrained \
  -v $(pwd)/output:/app/data/output \
  seed-story
```

## 🔧 Development

### Custom Configuration

```bash
# Mount custom configs
docker run --gpus all -p 7860:7860 \
  -v $(pwd)/custom-configs:/app/configs \
  seed-story
```

### Debugging

```bash
# Debug mode with verbose logging
docker run --gpus all -e SEED_STORY_MODE=cli -it seed-story bash

# Check model status
python3 model_downloader.py --list

# Test generation
python3 src/inference/gen_george.py
```

## 📊 Performance

### Recommended Hardware

| Configuration | GPU | VRAM | RAM | Story Length |
|---------------|-----|------|-----|--------------|
| **Minimum** | RTX 3080 | 16GB | 16GB | 5 sequences |
| **Recommended** | RTX 4090 | 24GB | 32GB | 10 sequences |
| **Optimal** | A100 | 40GB+ | 64GB+ | 25 sequences |

### Generation Times

- **Text Generation**: ~30-60 seconds per segment
- **Image Generation**: ~15-30 seconds per image (50 steps)
- **Total Story (10 segments)**: ~8-15 minutes

## 🛠️ Troubleshooting

### Common Issues

#### Out of Memory

```bash
# Reduce story length and image steps
# Use smaller batch sizes
# Close other GPU applications
```

#### Model Download Fails

```bash
# Check internet connection
# Ensure sufficient disk space
# Try manual download:
docker run --gpus all -e SEED_STORY_MODE=cli seed-story \
  python3 model_downloader.py --force
```

#### Permission Errors

```bash
# Fix data directory permissions
sudo chown -R $USER:$USER ./data
```

### Getting Help

1. **Check Logs**: Use `docker logs <container-id>` for error details
2. **System Info**: Access system status in web interface
3. **GitHub Issues**: Report bugs at [SEED-Story GitHub](https://github.com/TencentARC/SEED-Story/issues)

## 📚 Technical Details

### Architecture

SEED-Story uses a three-stage training approach:

1. **Visual Tokenization**: SDXL-based de-tokenizer for image reconstruction
2. **Instruction Tuning**: MLLM training with multimodal sequences
3. **De-tokenizer Adaptation**: Fine-tuning for character/style consistency

### Model Components

- **Visual Encoder**: Qwen-VL for image understanding
- **Language Model**: Llama-2-7B for text generation
- **Image Generator**: Stable Diffusion XL for image synthesis
- **Adapter**: Custom adapter for multimodal alignment

## 📖 Citation

```bibtex
@article{yang2024seedstory,
    title={SEED-Story: Multimodal Long Story Generation with Large Language Model},
    author={Shuai Yang and Yuying Ge and Yang Li and Yukang Chen and Yixiao Ge and Ying Shan and Yingcong Chen},
    year={2024},
    journal={arXiv preprint arXiv:2407.08683},
    url={https://arxiv.org/abs/2407.08683}
}
```

## 📄 License

This project follows the Apache License Version 2.0, consistent with the original SEED-Story implementation. See the original repository for detailed license information.

---

## 🔗 Related Projects

- **[YuE Music Generation](../yue/)** - AI music composition and generation
- **[SEED-X Multimodal](https://github.com/AILab-CVC/SEED-X)** - Foundation for SEED-Story
- **[StoryStream Dataset](https://huggingface.co/datasets/TencentARC/StoryStream)** - Training dataset

**Part of the Docker AI Models Collection** - Production-ready containers for cutting-edge AI models.
