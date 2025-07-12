#!/usr/bin/env python3
"""
SEED-Story Model Downloader
Automatically downloads required models for SEED-Story
"""

import os
import sys
import subprocess
import tempfile
from pathlib import Path
from huggingface_hub import snapshot_download, hf_hub_download
import torch

class ModelDownloader:
    def __init__(self):
        self.pretrained_dir = Path("/app/pretrained")
        self.pretrained_dir.mkdir(exist_ok=True)
        
        # Required models configuration
        self.models = {
            "stable-diffusion-xl-base-1.0": {
                "repo": "stabilityai/stable-diffusion-xl-base-1.0",
                "size": "~7GB",
                "description": "Stable Diffusion XL base model for image generation"
            },
            "Llama-2-7b-hf": {
                "repo": "meta-llama/Llama-2-7b-hf",
                "size": "~13GB", 
                "description": "Llama 2 7B base language model",
                "requires_auth": True
            },
            "Qwen-VL-Chat": {
                "repo": "Qwen/Qwen-VL-Chat",
                "size": "~10GB",
                "description": "Qwen Vision-Language chat model"
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
    
    def download_huggingface_model(self, repo_id, local_dir, description):
        """Download model from Hugging Face"""
        print(f"📥 Downloading {description}...")
        print(f"   Repository: {repo_id}")
        print(f"   Destination: {local_dir}")
        
        try:
            snapshot_download(
                repo_id=repo_id,
                local_dir=local_dir,
                local_dir_use_symlinks=False,
                resume_download=True
            )
            print(f"✅ Successfully downloaded {repo_id}")
            return True
        except Exception as e:
            print(f"❌ Failed to download {repo_id}: {e}")
            return False
    
    def download_seed_story_files(self):
        """Download specific SEED-Story checkpoint files"""
        print("📥 Downloading SEED-Story checkpoints...")
        
        model_info = self.models["SEED-Story-George"]
        success = True
        
        for file_path in model_info["files"]:
            local_path = self.pretrained_dir / file_path
            local_path.parent.mkdir(parents=True, exist_ok=True)
            
            try:
                print(f"   Downloading {file_path}...")
                hf_hub_download(
                    repo_id=model_info["repo"],
                    filename=file_path,
                    local_dir=self.pretrained_dir,
                    local_dir_use_symlinks=False,
                    resume_download=True
                )
                print(f"   ✅ Downloaded {file_path}")
            except Exception as e:
                print(f"   ❌ Failed to download {file_path}: {e}")
                success = False
        
        return success
    
    def extract_qwen_vit(self):
        """Extract Qwen VIT weights"""
        print("🔧 Extracting Qwen VIT weights...")
        
        try:
            # Change to app directory and run extraction script
            os.chdir("/app")
            result = subprocess.run([
                sys.executable, "src/tools/reload_qwen_vit.py"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ Qwen VIT extraction completed")
                return True
            else:
                print(f"❌ VIT extraction failed: {result.stderr}")
                return False
        except Exception as e:
            print(f"❌ VIT extraction error: {e}")
            return False
    
    def download_all_models(self, force=False):
        """Download all required models"""
        print("🎬 SEED-Story Model Downloader")
        print("=" * 50)
        
        self.check_disk_space()
        
        # Track download status
        download_status = {}
        
        # Download Stable Diffusion XL
        model_name = "stable-diffusion-xl-base-1.0"
        if not self.check_model_exists(model_name) or force:
            model_info = self.models[model_name]
            local_dir = self.pretrained_dir / model_name
            success = self.download_huggingface_model(
                model_info["repo"], local_dir, model_info["description"]
            )
            download_status[model_name] = success
        else:
            print(f"✅ {model_name} already exists")
            download_status[model_name] = True
        
        # Download Llama 2 (may require authentication)
        model_name = "Llama-2-7b-hf"
        if not self.check_model_exists(model_name) or force:
            model_info = self.models[model_name]
            local_dir = self.pretrained_dir / model_name
            
            if model_info.get("requires_auth"):
                print(f"⚠️  {model_name} requires Hugging Face authentication")
                print("   Please ensure you have access and are logged in with `huggingface-cli login`")
            
            success = self.download_huggingface_model(
                model_info["repo"], local_dir, model_info["description"]
            )
            download_status[model_name] = success
        else:
            print(f"✅ {model_name} already exists")
            download_status[model_name] = True
        
        # Download Qwen-VL-Chat
        model_name = "Qwen-VL-Chat"
        if not self.check_model_exists(model_name) or force:
            model_info = self.models[model_name]
            local_dir = self.pretrained_dir / model_name
            success = self.download_huggingface_model(
                model_info["repo"], local_dir, model_info["description"]
            )
            download_status[model_name] = success
        else:
            print(f"✅ {model_name} already exists")
            download_status[model_name] = True
        
        # Download SEED-Story checkpoints
        if not self.check_model_exists("SEED-Story-George") or force:
            success = self.download_seed_story_files()
            download_status["SEED-Story-George"] = success
        else:
            print("✅ SEED-Story checkpoints already exist")
            download_status["SEED-Story-George"] = True
        
        # Extract Qwen VIT if needed
        if not (self.pretrained_dir / "qwen_vit_g.pth").exists():
            vit_success = self.extract_qwen_vit()
            download_status["Qwen-VIT-Extraction"] = vit_success
        
        # Summary
        print("\n" + "=" * 50)
        print("📊 Download Summary:")
        for model, status in download_status.items():
            status_icon = "✅" if status else "❌"
            print(f"   {status_icon} {model}")
        
        all_success = all(download_status.values())
        if all_success:
            print("\n🎉 All models downloaded successfully!")
            print("   SEED-Story is ready to use!")
        else:
            print("\n⚠️  Some downloads failed. Check the logs above.")
            print("   You may need to manually download missing models.")
        
        return all_success
    
    def list_models(self):
        """List all required models and their status"""
        print("📋 Required Models for SEED-Story:")
        print("=" * 60)
        
        total_size = 0
        for model_name, model_info in self.models.items():
            exists = self.check_model_exists(model_name)
            status = "✅ Downloaded" if exists else "❌ Missing"
            size = model_info.get("size", "Unknown")
            description = model_info.get("description", "")
            
            print(f"{status} {model_name}")
            print(f"        Size: {size}")
            print(f"        Description: {description}")
            print()
        
        print(f"Total estimated storage: ~32GB")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="SEED-Story Model Downloader")
    parser.add_argument("--list", action="store_true", help="List required models")
    parser.add_argument("--force", action="store_true", help="Force re-download existing models")
    parser.add_argument("--model", type=str, help="Download specific model only")
    
    args = parser.parse_args()
    
    downloader = ModelDownloader()
    
    if args.list:
        downloader.list_models()
    else:
        downloader.download_all_models(force=args.force)

if __name__ == "__main__":
    main()
