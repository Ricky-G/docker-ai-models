---
name: 🐛 Bug Report
about: Report a bug or issue with an existing AI model container
title: "[MODEL_NAME] Bug: [Brief Description]"
labels: ["bug"]
assignees: []
---

## 🐛 Bug Description

**Model Affected**: <!-- e.g., SEED-Story, YuE, etc. -->

**Brief Description**: 
<!-- A clear and concise description of what the bug is -->

## 🔄 Steps to Reproduce

1. <!-- First step -->
2. <!-- Second step -->
3. <!-- Third step -->
4. <!-- See error -->

**Docker Command Used**:
```bash
# Paste the exact command you used
docker run --gpus all -p 7860:7860 ...
```

## 🎯 Expected Behavior

<!-- A clear description of what you expected to happen -->

## 💥 Actual Behavior

<!-- A clear description of what actually happened -->

## 🖥️ Environment Information

**System**:
- OS: <!-- e.g., Windows 11, Ubuntu 22.04 -->
- Docker Version: <!-- Run: docker --version -->
- GPU: <!-- e.g., RTX 3080, RTX 4090 -->
- VRAM: <!-- e.g., 16GB -->

**Container Information**:
- Image Built Successfully: [ ] Yes [ ] No
- Container Starts: [ ] Yes [ ] No
- Web Interface Accessible: [ ] Yes [ ] No [ ] N/A

## 📋 Error Logs

**Docker Build Logs** (if build fails):
```
# Paste build error logs here
```

**Container Runtime Logs**:
```bash
# Run: docker logs [container-name]
# Paste the output here
```

**Browser Console Errors** (if web interface issue):
```
# Open browser dev tools and paste any errors
```

## 🔍 Additional Context

<!-- Add any other context about the problem here -->

**Screenshots**: 
<!-- If applicable, add screenshots to help explain your problem -->

**Workaround Found**: 
<!-- If you found a temporary fix, please share it -->

## ✅ Troubleshooting Attempted

- [ ] Rebuilt Docker image
- [ ] Cleared Docker cache (`docker system prune`)
- [ ] Checked available disk space (20GB+ free)
- [ ] Verified GPU drivers are up to date
- [ ] Tried with different Docker volume paths
- [ ] Checked firewall/antivirus settings

## 🚨 Urgency Level

- [ ] Critical (blocking all usage)
- [ ] High (major feature broken)
- [ ] Medium (workaround available)
- [ ] Low (minor inconvenience)
