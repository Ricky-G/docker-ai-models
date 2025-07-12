#!/usr/bin/env python3
"""
Verification script for SEED-Story Docker setup
Checks that all dependencies and configurations are correct
"""

import sys
import os
from pathlib import Path

def verify_setup():
    """Verify the SEED-Story setup"""
    print("🔍 Verifying SEED-Story Docker Setup...")
    print("=" * 50)
    
    issues = []
    warnings = []
    
    # 1. Check Python version
    print(f"✓ Python version: {sys.version}")
    
    # 2. Check critical imports
    print("\n📦 Checking dependencies...")
    try:
        import torch
        print(f"✓ PyTorch: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            warnings.append("⚠️  No CUDA GPU detected - will run in CPU mode")
    except ImportError as e:
        issues.append(f"❌ PyTorch import failed: {e}")
    
    try:
        import gradio as gr
        print(f"✓ Gradio: {gr.__version__}")
    except ImportError as e:
        issues.append(f"❌ Gradio import failed: {e}")
    
    try:
        import diffusers
        print(f"✓ Diffusers: {diffusers.__version__}")
    except ImportError as e:
        issues.append(f"❌ Diffusers import failed: {e}")
    
    try:
        import transformers
        print(f"✓ Transformers: {transformers.__version__}")
    except ImportError as e:
        issues.append(f"❌ Transformers import failed: {e}")
    
    # 3. Check model directories
    print("\n📁 Checking directories...")
    models_dir = Path(os.environ.get('SEED_STORY_MODELS_DIR', '/app/pretrained'))
    
    if models_dir.exists():
        print(f"✓ Models directory exists: {models_dir}")
        
        # Check for specific models
        sdxl_path = models_dir / "stable-diffusion-xl-base-1.0"
        if sdxl_path.exists():
            print(f"✓ SDXL model found: {sdxl_path}")
        else:
            warnings.append(f"⚠️  SDXL model not found at {sdxl_path}")
    else:
        warnings.append(f"⚠️  Models directory not found: {models_dir}")
    
    # 4. Check environment variables
    print("\n🔧 Environment variables:")
    print(f"  SEED_STORY_MODE: {os.environ.get('SEED_STORY_MODE', 'not set')}")
    print(f"  SEED_STORY_MODELS_DIR: {os.environ.get('SEED_STORY_MODELS_DIR', 'not set')}")
    print(f"  GRADIO_SERVER_NAME: {os.environ.get('GRADIO_SERVER_NAME', 'not set')}")
    print(f"  GRADIO_SERVER_PORT: {os.environ.get('GRADIO_SERVER_PORT', 'not set')}")
    
    # 5. Check Python scripts
    print("\n📄 Checking Python scripts...")
    scripts = [
        "minimal_gradio.py",
        "simple_comic_generator.py",
        "model_downloader.py"
    ]
    
    for script in scripts:
        script_path = Path("/app") / script
        if script_path.exists():
            print(f"✓ {script} found")
            
            # Try to compile the script to check for syntax errors
            try:
                with open(script_path, 'r') as f:
                    compile(f.read(), script, 'exec')
                print(f"  ✓ No syntax errors in {script}")
            except SyntaxError as e:
                issues.append(f"❌ Syntax error in {script}: {e}")
        else:
            issues.append(f"❌ Script not found: {script_path}")
    
    # Report results
    print("\n" + "=" * 50)
    print("📊 VERIFICATION RESULTS:")
    
    if issues:
        print(f"\n❌ Found {len(issues)} critical issues:")
        for issue in issues:
            print(f"  {issue}")
    
    if warnings:
        print(f"\n⚠️  Found {len(warnings)} warnings:")
        for warning in warnings:
            print(f"  {warning}")
    
    if not issues and not warnings:
        print("\n✅ All checks passed! Your SEED-Story setup is ready.")
    elif not issues:
        print("\n✅ Setup is functional with minor warnings.")
    else:
        print("\n❌ Critical issues found. Please fix them before running.")
        sys.exit(1)

if __name__ == "__main__":
    verify_setup()