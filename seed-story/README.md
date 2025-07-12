# 🎬 SEED-## 🌟 What This Does

Enter a text prompt like "a cat exploring space" and get:

- 📖 **Generated Story Text**: Multi-panel narrative with consistent plot
- 🎨 **Comic Images**: AI-generated illustrations using Stable Diffusion XL
- 🖥️ **Web Interface**: Beautiful Gradio UI accessible via browser
- 🔄 **Customizable**: Adjust number of panels (1-8) and story themes

![Comic Example](https://via.placeholder.com/600x300/4CAF50/white?text=Comic+Story+Panel+Example)

## 🚀 Quick Start

### Prerequisites

- **NVIDIA GPU** with 8GB+ VRAM (16GB+ recommended)
- **Docker** with NVIDIA Container Runtime
- **~20GB** free disk space for models

### 1. Build the Container

```bash
git clone https://github.com/Ricky-G/docker-ai-models.gitr

cd docker-ai-models/seed-story
docker build -t seed-story .
```

### 2. Run with GPU Support

```bash
# Create models directory first
mkdir -p C:\_Models\seed-story

# Run with persistent model caching
docker run --name seed-story-container \
  --gpus all \
  -p 7860:7860 \
  -v "C:\_Models\seed-story:/app/models" \
  -v "C:\_Models\seed-story\pip-cache:/root/.cache/pip" \
  seed-story
```

### 3. Access the Web Interface

Open your browser and go to: <http://localhost:7860>

**That's it!** 🎉 Enter a prompt and start generating comics!

## 📋 How It Works

### AI Models Used

| Model | Size | Purpose |
|-------|------|---------|
| **Stable Diffusion XL** | ~7GB | Generates comic panel images |
| **Llama-2-7B** | ~13GB | Creates story narratives |
| **Qwen-VL-Chat** | ~10GB | Vision-language understanding |

**Total Storage**: ~20GB (models auto-download on first run)

### Generation Process

1. **Input**: You enter a text prompt (e.g., "space adventure")
2. **Story Creation**: AI generates a multi-panel narrative
3. **Image Generation**: Each panel gets a matching AI-generated image
4. **Display**: View your complete comic story in the web interface

## 🎯 Example Usage

### Web Interface Steps

1. Open <http://localhost:7860> in your browser
2. Enter a story prompt like:
   - "a cat discovering magic"
   - "robot exploring ancient ruins"
   - "wizard learning spells"
3. Choose number of panels (1-8)
4. Click "Generate Comic Story"
5. Watch as your story unfolds with images!

### Sample Prompts

- **Adventure**: "brave explorer finds hidden treasure"
- **Fantasy**: "young wizard saves magical kingdom" 
- **Sci-Fi**: "astronaut discovers alien civilization"
- **Mystery**: "detective solves impossible crime"

## ⚙️ Configuration

### Environment Variables

```bash
# Run on different port
docker run -p 8080:7860 -e GRADIO_SERVER_PORT=7860 seed-story

# Use specific GPU
docker run --gpus '"device=1"' seed-story
```

### Storage Options

```bash
# Mount custom model directory
docker run -v "/your/models/path:/app/models" seed-story

# Mount output directory for saving comics
docker run -v "./comics:/app/output" seed-story
```

## 🔧 Troubleshooting

### Common Issues

**Container won't start:**

```bash
# Check if container exists
docker ps -a

# View logs
docker logs seed-story-container

# Remove and rebuild
docker rm seed-story-container
docker build -t seed-story .
```

**Out of GPU memory:**

- Reduce number of panels to 1-3
- Close other GPU applications
- Use smaller batch sizes

**Models not downloading:**

- Check internet connection
- Ensure sufficient disk space (20GB+)
- Verify volume mount paths

### Getting Help

1. **Check Logs**: `docker logs seed-story-container`
2. **System Status**: Available in web interface
3. **Issues**: Report at [GitHub Issues](https://github.com/Ricky-G/docker-ai-models/issues)

## 📊 Performance

### Recommended Hardware

| Configuration | GPU | VRAM | Generation Time |
|---------------|-----|------|-----------------|
| **Minimum** | RTX 3060 | 8GB | ~5 min/story |
| **Recommended** | RTX 3080 | 16GB | ~3 min/story |
| **Optimal** | RTX 4090 | 24GB | ~1 min/story |

### Speed Tips

- Use fewer panels for faster generation
- Keep other GPU applications closed
- Use SSD storage for models
- Ensure good cooling for sustained generation

## 🛠️ Development

### File Structure

```
seed-story/
├── dockerfile          # Container definition
├── startup.sh         # Container startup script
├── minimal_gradio.py   # Main web interface
├── simple_comic_generator.py  # Fallback generator
├── model_downloader.py # Model management
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

### Custom Modifications

```bash
# Run in development mode
docker run -it --gpus all -v $(pwd):/app/dev seed-story bash

# Test individual components
python minimal_gradio.py
python model_downloader.py --list
```

## 📚 Based On

This Docker implementation is built on:

- **[SEED-Story](https://github.com/TencentARC/SEED-Story)** by TencentARC
- **[Stable Diffusion XL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)** for image generation
- **[Llama-2](https://huggingface.co/meta-llama/Llama-2-7b-hf)** for text generation
- **[Gradio](https://gradio.app/)** for the web interface

## 📄 License

This project follows the licenses of its component models. See original repositories for detailed license information.

---

**Part of the [Docker AI Models Collection](https://github.com/Ricky-G/docker-ai-models)** - Production-ready containers for cutting-edge AI models.
