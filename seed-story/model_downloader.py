#!/usr/bin/env python3
"""
SEED-Story Model Downloader
Automatically downloads required models for SEED-Story
"""

import os
import sys
import subprocess
from pathlib import Path
from huggingface_hub import snapshot_download, hf_hub_download
import torch

class ModelDownloader:
    def __init__(self):
        # Use mounted volume if available, otherwise use default container directory
        models_dir = os.environ.get('SEED_STORY_MODELS_DIR', '/app/pretrained')
        self.pretrained_dir = Path(models_dir)
        self.pretrained_dir.mkdir(exist_ok=True)
        
        print(f"📁 Models will be downloaded to: {self.pretrained_dir}")
        
        # Required models configuration with optimized file patterns
        self.models = {
            "stable-diffusion-xl-base-1.0": {
                "repo": "stabilityai/stable-diffusion-xl-base-1.0",
                "size": "~7GB",
                "description": "Stable Diffusion XL base model for image generation",
                "allow_patterns": [
                    "*.json", "*.bin", "*.safetensors",
                    "**/model_index.json", "**/config.json", "**/pytorch_model*.bin",
                    "**/diffusion_pytorch_model.safetensors", "**/diffusion_pytorch_model.bin",
                    "scheduler/**", "unet/**", "vae/**",
                    "text_encoder/**", "text_encoder_2/**",
                    "tokenizer/**", "tokenizer_2/**",
                    "README.md"
                ]
            },
            "Llama-2-7b-hf": {
                "repo": "huggyllama/llama-7b",
                "size": "~13GB",
                "description": "Llama 7B base language model (HuggingFace converted)",
                "requires_auth": False,
                "allow_patterns": [
                    "config.json", "generation_config.json",
                    "pytorch_model*.bin", "model*.safetensors",
                    "tokenizer.model", "tokenizer.json", "tokenizer_config.json",
                    "special_tokens_map.json", "README.md"
                ]
            },
            "Qwen-VL-Chat": {
                "repo": "Qwen/Qwen-VL-Chat",
                "size": "~10GB",
                "description": "Qwen Vision-Language chat model",
                "allow_patterns": [
                    "config.json", "generation_config.json",
                    "pytorch_model*.bin", "model*.safetensors",
                    "tokenizer.model", "tokenizer.json", "tokenizer_config.json",
                    "qwen.tiktoken", "special_tokens_map.json",
                    "visual.py", "modeling_qwen.py", "tokenization_qwen.py", "configuration_qwen.py",
                    "README.md"
                ]
            },
            "SEED-Story-George": {
                "repo": "TencentARC/SEED-Story",
                "files": [
                    "SEED-Story-George/pytorch_model.bin",
                    "Tokenizer/tokenizer.model",
                    "Detokenizer-George/pytorch_model.bin"
                ],
                "size": "~2GB",
                "description": "SEED-Story pre-trained checkpoints"
            }
        }

    def check_disk_space(self):
        """Check available disk space"""
        stat = os.statvfs(self.pretrained_dir)
        free_space_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)
        print(f"💾 Available disk space: {free_space_gb:.1f}GB")
        
        if free_space_gb < 35:
            print("⚠️  Warning: Less than 35GB free space available")
            print("   SEED-Story requires ~32GB for all models")
            response = input("Continue anyway? (y/N): ")
            if response.lower() != 'y':
                sys.exit(1)

    def check_model_exists(self, model_name):
        """Check if model already exists"""
        model_path = self.pretrained_dir / model_name
        return model_path.exists() and any(model_path.iterdir())

    def download_huggingface_model(self, repo_id, local_dir, description, allow_patterns=None, ignore_patterns=None):
        """Download model from Hugging Face with proper caching"""
        print(f"📥 Downloading {description}...")
        print(f"   Repository: {repo_id}")
        print(f"   Destination: {local_dir}")
        
        try:
            # Set up environment for proper caching
            import os
            
            # Ensure cache directory exists
            cache_dir = self.pretrained_dir / "huggingface" / "hub"
            cache_dir.mkdir(parents=True, exist_ok=True)
            
            # Set HF cache environment
            os.environ['HUGGINGFACE_HUB_CACHE'] = str(cache_dir)
            os.environ['HF_HOME'] = str(self.pretrained_dir / "huggingface")
            
            # Download using snapshot_download for complete model
            downloaded_path = snapshot_download(
                repo_id=repo_id,
                cache_dir=str(cache_dir),
                allow_patterns=allow_patterns,
                ignore_patterns=ignore_patterns,
                resume_download=True,
                local_files_only=False
            )
            
            # Create symlink to expected location for backward compatibility
            if not local_dir.exists():
                try:
                    local_dir.symlink_to(downloaded_path)
                except OSError:
                    # If symlinks aren't supported, copy essential files
                    local_dir.mkdir(parents=True, exist_ok=True)
                    import shutil
                    for item in Path(downloaded_path).rglob("*"):
                        if item.is_file():
                            rel_path = item.relative_to(downloaded_path)
                            dest_path = local_dir / rel_path
                            dest_path.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(item, dest_path)
            
            print(f"✅ Successfully downloaded {description}")
            print(f"   Cached at: {downloaded_path}")
            print(f"   Linked to: {local_dir}")
            return True
        except Exception as e:
            print(f"❌ Failed to download {description}: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def download_all_models(self):
        """Download all required models"""
        print("🚀 Starting model downloads...")
        self.check_disk_space()
        
        # Define models with optimized patterns (only essential files)
        models = {
            "stable-diffusion-xl-base-1.0": {
                "repo_id": "stabilityai/stable-diffusion-xl-base-1.0",
                "description": "Stable Diffusion XL Base",
                "allow_patterns": [
                    "model_index.json",
                    "scheduler/scheduler_config.json",
                    "text_encoder/config.json",
                    "text_encoder/pytorch_model.bin",
                    "text_encoder_2/config.json", 
                    "text_encoder_2/pytorch_model.bin",
                    "tokenizer/tokenizer_config.json",
                    "tokenizer/vocab.json",
                    "tokenizer/merges.txt",
                    "tokenizer_2/tokenizer_config.json",
                    "tokenizer_2/vocab.json", 
                    "tokenizer_2/merges.txt",
                    "unet/config.json",
                    "unet/diffusion_pytorch_model.safetensors",
                    "vae/config.json",
                    "vae/diffusion_pytorch_model.safetensors"
                ],
                "ignore_patterns": ["*.onnx", "*.trt", "*.engine", "*.pb", "*fp16*", "*flax*"]
            },
            "Llama-2-7b-hf": {
                "repo_id": "huggyllama/llama-7b",
                "description": "Llama 7B (HuggingFace converted)",
                "allow_patterns": [
                    "config.json",
                    "generation_config.json",
                    "pytorch_model-*.bin",
                    "tokenizer.model",
                    "tokenizer_config.json",
                    "special_tokens_map.json"
                ],
                "ignore_patterns": ["*.onnx", "*.trt", "*.engine", "*.pb", "*onnx*", "*tensorrt*", "*tf_*"]
            },
            "Qwen-VL-Chat": {
                "repo_id": "Qwen/Qwen-VL-Chat",
                "description": "Qwen Vision-Language Model", 
                "allow_patterns": [
                    "config.json",
                    "generation_config.json",
                    "pytorch_model-*.bin",
                    "tokenizer.json",
                    "qwen.tiktoken",
                    "*.py"
                ],
                "ignore_patterns": ["*.onnx", "*.trt", "*.engine", "*.pb", "*onnx*", "*tensorrt*", "*tf_*"]
            }
        }
        
        success_count = 0
        for model_name, config in models.items():
            if not self.check_model_exists(model_name):
                print(f"\n📦 Downloading {model_name}...")
                local_path = self.pretrained_dir / model_name
                
                success = self.download_huggingface_model(
                    repo_id=config["repo_id"],
                    local_dir=str(local_path),
                    description=config["description"],
                    allow_patterns=config.get("allow_patterns"),
                    ignore_patterns=config.get("ignore_patterns")
                )
                
                if success:
                    success_count += 1
                else:
                    print(f"⚠️  Failed to download {model_name}")
            else:
                print(f"✅ {model_name} already exists, skipping")
                success_count += 1
        
        print(f"\n🎉 Download completed! {success_count}/{len(models)} models ready")
        return success_count == len(models)

if __name__ == "__main__":
    print("🤖 SEED-Story Model Downloader")
    print("=" * 50)
    
    downloader = ModelDownloader()
    success = downloader.download_all_models()
    
    if success:
        print("✅ All models downloaded successfully!")
        sys.exit(0)
    else:
        print("❌ Some models failed to download")
        sys.exit(1)