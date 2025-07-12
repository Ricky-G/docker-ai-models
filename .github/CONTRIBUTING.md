# Contributing to docker-ai-models

Thank you for your interest in contributing! This repository aims to make cutting-edge AI models accessible through simple Docker containers.

## 🎯 What We're Looking For

**New AI Model Containers**:
- Image generation (Stable Diffusion variants, etc.)
- Text generation (Language models, chat bots)
- Audio/Music generation and processing
- Video generation and processing
- Multimodal models (text+image, audio+visual)

**Improvements to Existing Models**:
- Performance optimizations
- Better error handling
- Enhanced web interfaces
- Documentation improvements
- Support for different hardware configurations

## 🚀 Quick Start for Contributors

### 1. Fork and Clone
```bash
git clone https://github.com/YOUR_USERNAME/docker-ai-models.git
cd docker-ai-models
```

### 2. Create Your Model Directory
```bash
mkdir your-model-name/
cd your-model-name/
```

### 3. Required Files Structure
```
your-model-name/
├── dockerfile              # Main container build file
├── startup.sh             # Container startup script  
├── README.md              # Setup and usage guide
├── requirements.txt       # Python dependencies
├── gradio_interface.py    # Web UI (if applicable)
└── dockerfile-gpu-poor    # Optional: low-VRAM variant
```

## 📋 Standards and Guidelines

### Docker Requirements
- Use multi-stage builds when possible
- Include health checks for web services
- Optimize for layer caching
- Use specific base image versions (not `latest`)
- Support GPU passthrough with `--gpus all`

### Python Code Standards
- Include type hints and docstrings
- Use proper error handling with user-friendly messages
- Disable Gradio analytics: `os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"`
- Include progress indicators for long operations
- Validate inputs and provide helpful defaults

### Documentation Requirements
- Clear hardware requirements (GPU VRAM, storage)
- Step-by-step setup instructions
- Example usage with real commands
- Troubleshooting section
- Performance benchmarks (if available)

### Web Interface Guidelines
- Use Gradio for consistency across models
- Include example inputs and clear instructions
- Add progress bars for generation tasks
- Show system status (GPU, model loading, etc.)
- Handle errors gracefully with helpful messages

## 🔧 Development Process

### 1. Test Your Container
```bash
# Build the container
docker build -t your-model-name .

# Test with GPU
docker run --gpus all -p 7860:7860 your-model-name

# Verify web interface at http://localhost:7860
```

### 2. Validate Requirements
- [ ] Container builds without errors
- [ ] Models download successfully on first run
- [ ] Web interface is responsive and functional
- [ ] Basic inference/generation works
- [ ] Error handling provides helpful messages
- [ ] Documentation is complete and clear

### 3. Submit Pull Request
Use our [PR template](.github/pull_request_template.md) which includes:
- Model information and requirements
- Testing checklist
- Example usage
- Documentation quality verification

## 🎨 Example Model Implementation

### Basic Dockerfile Structure
```dockerfile
FROM nvcr.io/nvidia/pytorch:23.10-py3

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git wget curl && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . /app/
RUN chmod +x startup.sh

# Expose port and set startup command
EXPOSE 7860
CMD ["./startup.sh"]
```

### Basic Startup Script
```bash
#!/bin/bash
set -e

echo "🔧 Starting [MODEL_NAME]..."

# Check GPU availability
if ! nvidia-smi > /dev/null 2>&1; then
    echo "❌ No GPU detected. This model requires CUDA GPU."
    exit 1
fi

# Download models if needed
python model_downloader.py

# Start web interface
echo "🌐 Starting web interface on http://localhost:7860"
python gradio_interface.py
```

### Basic Gradio Interface
```python
import gradio as gr
import os

# Disable analytics
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

def generate(prompt):
    try:
        # Your model inference code here
        result = your_model.generate(prompt)
        return result
    except Exception as e:
        return f"❌ Error: {str(e)}"

# Create interface
interface = gr.Interface(
    fn=generate,
    inputs=gr.Textbox(label="Prompt", placeholder="Enter your prompt..."),
    outputs=gr.Textbox(label="Result"),
    title="🤖 Your Model Name",
    description="Brief description of what your model does"
)

if __name__ == "__main__":
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_api=False
    )
```

## 🛠️ Testing Guidelines

### Local Testing
1. Build container from scratch
2. Test with clean Docker cache
3. Verify model downloads work
4. Test basic functionality
5. Check error handling scenarios
6. Validate documentation accuracy

### Hardware Testing
- Test on minimum specified GPU VRAM
- Verify performance on recommended hardware
- Test with different Docker configurations
- Ensure container works on both Windows and Linux

## 📚 Resources

### Useful Base Images
- `nvcr.io/nvidia/pytorch:23.10-py3` - PyTorch with CUDA
- `tensorflow/tensorflow:2.13.0-gpu` - TensorFlow with GPU
- `huggingface/transformers-pytorch-gpu` - HuggingFace models

### Common Dependencies
- `torch` - PyTorch for most AI models
- `transformers` - HuggingFace transformers
- `diffusers` - Stable Diffusion and variants
- `gradio` - Web interface framework
- `Pillow` - Image processing
- `numpy` - Numerical computations

### Model Sources
- [HuggingFace Hub](https://huggingface.co/models)
- [GitHub trending AI repos](https://github.com/topics/artificial-intelligence)
- [Papers with Code](https://paperswithcode.com/)

## 🤝 Getting Help

- **Questions**: Open a [discussion](https://github.com/Ricky-G/docker-ai-models/discussions)
- **Bugs**: Use our [bug report template](.github/ISSUE_TEMPLATE/bug_report.md)
- **Features**: Use our [feature request template](.github/ISSUE_TEMPLATE/feature_request.md)

## 📝 Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Help others learn and contribute
- Focus on the value to the community

## 🏆 Recognition

Contributors will be:
- Listed in the repository contributors
- Credited in model-specific README files
- Mentioned in release notes for significant contributions

Thank you for helping make AI models more accessible! 🚀
