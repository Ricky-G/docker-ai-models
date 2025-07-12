# GitHub Copilot Instructions for docker-ai-models

## Repository Purpose
This repository provides **ready-to-run Docker containers** for cutting-edge AI models. The goal is to eliminate dependency hell and complex setup processes, allowing users to run AI models with simple Docker commands.

## Code Style & Standards

### Docker Best Practices
- Use multi-stage builds for smaller images
- Leverage Docker BuildKit with cache mounts
- Include health checks for long-running containers
- Use specific base image tags (avoid `latest`)
- Optimize layer caching with smart COPY ordering

### File Structure Requirements
Each AI model directory must contain:
```
model-name/
├── dockerfile                    # Main container build
├── dockerfile-gpu-poor          # Optional: low-VRAM variant
├── startup.sh                   # Container startup script
├── startup-gpu-poor.sh         # Optional: low-VRAM startup
├── README.md                    # Setup and usage guide
├── requirements.txt             # Python dependencies
└── gradio_interface.py          # Web UI (if applicable)
```

### Python Code Standards
- Use type hints for function parameters and returns
- Include comprehensive error handling with user-friendly messages
- Add progress indicators for long-running operations
- Validate environment variables and provide defaults
- Use pathlib for file operations
- Include GPU memory management and cleanup

### Gradio Interface Guidelines
- Disable analytics: `os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"`
- Use descriptive labels and helpful placeholders
- Include example inputs and clear instructions
- Add progress indicators for generation tasks
- Provide system status information (GPU, models, etc.)
- Handle errors gracefully with user-friendly messages

### Documentation Standards
- Write clear, step-by-step setup instructions
- Include hardware requirements (GPU VRAM, storage)
- Provide example commands with actual parameters
- Add troubleshooting sections for common issues
- Include performance benchmarks and optimization tips

### Environment Variables
Standard variables across all models:
```bash
CUDA_VISIBLE_DEVICES=0           # GPU selection
GRADIO_SERVER_NAME=0.0.0.0       # Web interface host
GRADIO_SERVER_PORT=7860          # Web interface port
MODEL_CACHE_DIR=/app/models      # Model storage location
```

### Error Handling Patterns
```python
try:
    # Model loading/generation code
    print("✅ Operation successful")
except torch.cuda.OutOfMemoryError:
    print("❌ GPU out of memory. Try reducing batch size or image resolution.")
except Exception as e:
    print(f"❌ Error: {e}")
    # Provide helpful troubleshooting steps
```

### Logging Standards
Use consistent emoji prefixes:
- 🔧 Setup/initialization
- 🟢 Success messages
- ⚠️ Warnings
- ❌ Errors
- 🎨 Generation progress
- 📁 File operations
- 🌐 Network/web interface

## AI Model Integration Guidelines

### Model Loading
- Check for model existence before loading
- Provide clear download progress indicators
- Support both local and remote model loading
- Handle model format conversions gracefully
- Implement model caching strategies

### GPU Management
- Detect GPU availability and capabilities
- Provide fallback for CPU-only systems
- Implement memory cleanup after operations
- Support multiple GPU configurations
- Include VRAM usage monitoring

### Web Interface Design
- Use intuitive parameter controls (sliders, dropdowns)
- Provide real-time generation progress
- Include model status indicators
- Add example prompts and use cases
- Implement result download/sharing options

## Security Considerations
- Sanitize user inputs before processing
- Avoid executing arbitrary code from user inputs
- Use safe file path handling
- Implement resource limits and timeouts
- Validate model checksums when downloading

## Performance Optimization
- Use appropriate data types (float16 vs float32)
- Implement batch processing where possible
- Cache compiled models and tokenizers
- Optimize Docker layer sizes
- Use efficient image formats and compression

## Testing Guidelines
- Include basic smoke tests for container builds
- Test with different GPU configurations
- Verify model downloads and loading
- Test web interface functionality
- Include example generation tests

## Common Patterns to Follow

### Startup Script Structure
```bash
#!/bin/bash
set -e

echo "🔧 Starting [MODEL_NAME]..."
check_gpu()
check_models()
setup_environment()
launch_interface()
```

### Python Main Function
```python
if __name__ == "__main__":
    try:
        print("🔧 Starting [MODEL_NAME] interface...")
        interface = create_interface()
        interface.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            show_api=False
        )
    except Exception as e:
        print(f"❌ Failed to start: {e}")
        sys.exit(1)
```

## Contributing Focus Areas
When contributing to this repository, prioritize:
1. **User Experience**: Make setup as simple as possible
2. **Hardware Compatibility**: Support various GPU configurations
3. **Clear Documentation**: Write for users new to AI/Docker
4. **Error Recovery**: Provide helpful error messages and solutions
5. **Performance**: Optimize for speed and memory efficiency

## Repository Goals
- **Zero Setup Complexity**: One command to run any AI model
- **Hardware Flexibility**: Support from 8GB to 24GB+ VRAM
- **Comprehensive Coverage**: Include diverse AI model types
- **Production Quality**: Stable, tested, and documented containers
- **Community Driven**: Easy for others to contribute new models
