---
name: 🤖 AI Model Container
about: Add a new AI model container to the repository
title: "Add [MODEL_NAME] - [Brief Description]"
labels: ["enhancement", "new-model"]
assignees: []
---

## 🎯 Model Information

**Model Name**: 
**Model Type**: <!-- e.g., Image Generation, Text-to-Speech, Video Generation -->
**Original Repository**: <!-- Link to original model repository -->
**License**: <!-- Model license (MIT, Apache, etc.) -->

## 📋 Container Details

**Hardware Requirements**:
- **Minimum GPU VRAM**: <!-- e.g., 8GB -->
- **Recommended GPU VRAM**: <!-- e.g., 16GB -->
- **Storage Required**: <!-- e.g., ~15GB for models -->

**Interface Type**:
- [ ] Web UI (Gradio)
- [ ] CLI only
- [ ] Both Web UI and CLI

## 📁 Files Included

Please confirm you've included all required files:

- [ ] `dockerfile` - Main container build file
- [ ] `startup.sh` - Container startup script
- [ ] `README.md` - Setup and usage documentation
- [ ] `requirements.txt` - Python dependencies
- [ ] Web interface file (if applicable)
- [ ] `dockerfile-gpu-poor` (if low-VRAM variant available)

## ✅ Testing Checklist

- [ ] Docker image builds successfully
- [ ] Container starts without errors
- [ ] Models download correctly on first run
- [ ] Web interface is accessible (if applicable)
- [ ] Basic generation/inference works
- [ ] Tested on GPU with specified VRAM requirements
- [ ] Error handling works (e.g., missing models, GPU issues)

## 🎨 Example Usage

**Sample Input**: 
<!-- Provide an example of what users would input -->

**Expected Output**: 
<!-- Describe what the model should generate -->

**Docker Command**:
```bash
# Build command
docker build -t [model-name] .

# Run command
docker run --gpus all -p 7860:7860 -v "path/to/models:/app/models" [model-name]
```

## 📖 Documentation Quality

- [ ] README includes clear setup instructions
- [ ] Hardware requirements are specified
- [ ] Example prompts/inputs are provided
- [ ] Troubleshooting section is included
- [ ] Performance tips are mentioned
- [ ] Docker commands are tested and correct

## 🔧 Technical Implementation

**Base Image Used**: <!-- e.g., nvcr.io/nvidia/pytorch:23.10-py3 -->

**Key Dependencies**: 
<!-- List major Python packages or system dependencies -->

**GPU Framework**: 
- [ ] PyTorch
- [ ] TensorFlow
- [ ] JAX
- [ ] Other: ___________

## 🌟 Additional Features

- [ ] Multiple generation modes/styles
- [ ] Batch processing capability
- [ ] Custom parameter controls
- [ ] Export/download functionality
- [ ] Progress indicators
- [ ] Memory optimization options

## 📝 Additional Notes

<!-- Any special considerations, known issues, or additional context -->

## 🚀 Ready for Review

- [ ] I have tested this thoroughly
- [ ] Documentation is complete and clear
- [ ] Code follows repository standards
- [ ] All required files are included
- [ ] This works on the specified hardware requirements

---

**Note for Reviewers**: Please verify that the container builds and runs correctly, and that the documentation is clear for new users.
