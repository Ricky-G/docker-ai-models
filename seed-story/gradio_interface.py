#!/usr/bin/env python3
"""
SEED-Story Gradio Web Interface
Interactive web interface for multimodal story generation
"""

import gradio as gr
import torch
import os
import sys
import json
import tempfile
import shutil
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import logging
import traceback
import psutil
import threading
import time

# Add project root to Python path
sys.path.insert(0, '/app')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SEEDStoryInterface:
    """Main interface class for SEED-Story web UI"""
    
    def __init__(self):
        self.model_loaded = False
        self.models = {}
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.output_dir = Path("/app/data/output")
        self.output_dir.mkdir(exist_ok=True)
        
        # Generation parameters
        self.default_params = {
            "story_length": 10,
            "window_size": 8,
            "max_new_tokens": 500,
            "num_inference_steps": 50,
            "temperature": 0.7,
            "top_p": 0.9,
            "guidance_scale": 7.5
        }
        
        # Initialize model loading
        self.load_models()
    
    def load_models(self):
        """Load SEED-Story models"""
        try:
            logger.info("🔄 Loading SEED-Story models...")
            
            # Check if models exist
            pretrained_dir = Path("/app/pretrained")
            required_models = [
                "stable-diffusion-xl-base-1.0",
                "Llama-2-7b-hf", 
                "Qwen-VL-Chat"
            ]
            
            missing_models = []
            for model in required_models:
                if not (pretrained_dir / model).exists():
                    missing_models.append(model)
            
            if missing_models:
                logger.warning(f"Missing models: {missing_models}")
                return False
            
            # Import SEED-Story modules
            try:
                import hydra
                from omegaconf import OmegaConf
                import pyrootutils
                
                # Setup project root
                pyrootutils.setup_root('/app', indicator='.project-root', pythonpath=True)
                
                # Load configurations
                self.load_configurations()
                
                # Initialize components
                self.initialize_components()
                
                self.model_loaded = True
                logger.info("✅ SEED-Story models loaded successfully")
                return True
                
            except Exception as e:
                logger.error(f"Failed to load SEED-Story modules: {e}")
                return False
                
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False
    
    def load_configurations(self):
        """Load model configurations"""
        try:
            from omegaconf import OmegaConf
            
            # Configuration paths
            self.configs = {
                'tokenizer': 'configs/tokenizer/clm_llama_tokenizer.yaml',
                'image_transform': 'configs/processer/qwen_448_transform.yaml',
                'visual_encoder': 'configs/visual_tokenizer/qwen_vitg_448.yaml',
                'llm': 'configs/clm_models/llama2chat7b_lora.yaml',
                'agent': 'configs/clm_models/agent_7b_sft.yaml',
                'adapter': 'configs/detokenizer/detokenizer_sdxl_qwen_vit_adapted.yaml',
                'discrete_model': 'configs/discrete_model/discrete_identity.yaml'
            }
            
            # Load all configurations
            self.config_objects = {}
            for name, path in self.configs.items():
                config_path = Path('/app') / path
                if config_path.exists():
                    self.config_objects[name] = OmegaConf.load(config_path)
                else:
                    logger.warning(f"Config file not found: {config_path}")
            
        except Exception as e:
            logger.error(f"Error loading configurations: {e}")
    
    def initialize_components(self):
        """Initialize SEED-Story model components"""
        try:
            import hydra
            from diffusers import AutoencoderKL, UNet2DConditionModel, EulerDiscreteScheduler
            
            # Initialize tokenizer
            if 'tokenizer' in self.config_objects:
                self.models['tokenizer'] = hydra.utils.instantiate(self.config_objects['tokenizer'])
            
            # Initialize image transform
            if 'image_transform' in self.config_objects:
                self.models['image_transform'] = hydra.utils.instantiate(self.config_objects['image_transform'])
            
            # Initialize visual encoder
            if 'visual_encoder' in self.config_objects:
                self.models['visual_encoder'] = hydra.utils.instantiate(self.config_objects['visual_encoder'])
                self.models['visual_encoder'].eval().to(self.device, dtype=self.dtype)
            
            # Initialize LLM
            if 'llm' in self.config_objects:
                self.models['llm'] = hydra.utils.instantiate(
                    self.config_objects['llm'], 
                    torch_dtype='fp16' if self.dtype == torch.float16 else 'fp32'
                )
            
            # Initialize agent model
            if 'agent' in self.config_objects and 'llm' in self.models:
                self.models['agent'] = hydra.utils.instantiate(
                    self.config_objects['agent'], 
                    llm=self.models['llm']
                )
                self.models['agent'].eval().to(self.device, dtype=self.dtype)
            
            # Initialize diffusion components
            diffusion_path = "/app/pretrained/stable-diffusion-xl-base-1.0"
            if Path(diffusion_path).exists():
                self.models['scheduler'] = EulerDiscreteScheduler.from_pretrained(
                    diffusion_path, subfolder="scheduler"
                )
                self.models['vae'] = AutoencoderKL.from_pretrained(
                    diffusion_path, subfolder="vae"
                ).to(self.device, dtype=self.dtype)
                self.models['unet'] = UNet2DConditionModel.from_pretrained(
                    diffusion_path, subfolder="unet"
                ).to(self.device, dtype=self.dtype)
            
            # Initialize adapter
            if 'adapter' in self.config_objects and 'unet' in self.models:
                self.models['adapter'] = hydra.utils.instantiate(
                    self.config_objects['adapter'], 
                    unet=self.models['unet']
                ).to(self.device, dtype=self.dtype).eval()
                
                # Initialize adapter pipeline
                if all(k in self.models for k in ['vae', 'scheduler', 'visual_encoder', 'image_transform']):
                    discrete_model = hydra.utils.instantiate(
                        self.config_objects['discrete_model']
                    ).to(self.device).eval() if 'discrete_model' in self.config_objects else None
                    
                    self.models['adapter'].init_pipe(
                        vae=self.models['vae'],
                        scheduler=self.models['scheduler'],
                        visual_encoder=self.models['visual_encoder'],
                        image_transform=self.models['image_transform'],
                        discrete_model=discrete_model,
                        dtype=self.dtype,
                        device=self.device
                    )
            
            # Token IDs
            if 'tokenizer' in self.models:
                self.boi_token_id = self.models['tokenizer'].encode('<img>', add_special_tokens=False)[0]
                self.eoi_token_id = self.models['tokenizer'].encode('</img>', add_special_tokens=False)[0]
            
            logger.info("✅ Model components initialized")
            
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            logger.error(traceback.format_exc())
    
    def add_subtitle_to_image(self, image: Image.Image, text: str) -> Image.Image:
        """Add subtitle text to bottom of image"""
        try:
            text_height = 80
            new_size = (image.width, image.height + text_height)
            
            new_image = Image.new("RGB", new_size, "black")
            new_image.paste(image, (0, 0))
            
            draw = ImageDraw.Draw(new_image)
            font_size = 14
            
            # Split text into two lines
            mid_point = len(text) // 2
            line1 = text[:mid_point]
            line2 = text[mid_point:]
            
            y_pos = image.height + (text_height - font_size) // 2
            draw.text((10, y_pos), line1, fill="white")
            draw.text((10, y_pos + font_size), line2, fill="white")
            
            return new_image
            
        except Exception as e:
            logger.error(f"Error adding subtitle: {e}")
            return image
    
    def generate_story(
        self,
        input_image: Image.Image,
        initial_text: str,
        story_length: int = 10,
        max_new_tokens: int = 500,
        num_inference_steps: int = 50,
        temperature: float = 0.7,
        guidance_scale: float = 7.5,
        progress=gr.Progress()
    ) -> Tuple[List[Image.Image], List[str], str]:
        """Generate multimodal story"""
        
        if not self.model_loaded:
            return [], ["Error: Models not loaded"], "❌ Models not loaded. Please check model files."
        
        try:
            progress(0, desc="🔄 Initializing story generation...")
            
            # Create output directory for this generation
            output_dir = self.output_dir / f"story_{int(time.time())}"
            output_dir.mkdir(exist_ok=True)
            
            # Prepare input
            if input_image is None:
                return [], ["Error: No input image provided"], "❌ Please provide an input image"
            
            if not initial_text.strip():
                initial_text = "Once upon a time..."
            
            # Convert and process image
            if input_image.mode != 'RGB':
                input_image = input_image.convert('RGB')
            
            progress(0.1, desc="🖼️ Processing input image...")
            
            # Prepare image tokens
            BOI_TOKEN = '<img>'
            EOI_TOKEN = '</img>'
            IMG_TOKEN = '<img_{:05d}>'
            num_img_in_tokens = 64
            num_img_out_tokens = 64
            
            image_tokens = BOI_TOKEN + ''.join([
                IMG_TOKEN.format(int(item)) for item in range(num_img_in_tokens)
            ]) + EOI_TOKEN
            
            # Create prompt
            prompt = initial_text + image_tokens
            
            progress(0.2, desc="🧠 Encoding image features...")
            
            # Process image
            image_tensor = self.models['image_transform'](input_image).unsqueeze(0).to(self.device, dtype=self.dtype)
            
            with torch.no_grad():
                image_embeds = self.models['visual_encoder'](image_tensor)
            
            # Initialize story generation
            self.models['agent'].llm.base_model.model.use_kv_cache_head = False
            
            # Tokenize prompt
            input_ids = self.models['tokenizer'].encode(prompt, add_special_tokens=False)
            input_ids = [self.models['tokenizer'].bos_token_id] + input_ids
            
            boi_idx = input_ids.index(self.boi_token_id)
            eoi_idx = input_ids.index(self.eoi_token_id)
            
            input_ids = torch.tensor(input_ids).to(self.device, dtype=torch.long).unsqueeze(0)
            
            ids_cmp_mask = torch.zeros_like(input_ids, dtype=torch.bool)
            ids_cmp_mask[0, boi_idx + 1:eoi_idx] = True
            embeds_cmp_mask = torch.tensor([True]).to(self.device, dtype=torch.bool)
            
            # Generated content
            generated_images = []
            generated_texts = []
            
            # Add initial image with subtitle
            initial_with_subtitle = self.add_subtitle_to_image(input_image, initial_text)
            generated_images.append(initial_with_subtitle)
            generated_texts.append(initial_text)
            
            # Save initial image
            initial_path = output_dir / "00_initial.jpg"
            initial_with_subtitle.save(initial_path)
            
            progress(0.3, desc="📝 Generating first story segment...")
            
            # Generate first story segment
            with torch.no_grad():
                output = self.models['agent'].generate(
                    tokenizer=self.models['tokenizer'],
                    input_ids=input_ids,
                    image_embeds=image_embeds,
                    embeds_cmp_mask=embeds_cmp_mask,
                    ids_cmp_mask=ids_cmp_mask,
                    max_new_tokens=max_new_tokens,
                    num_img_gen_tokens=num_img_out_tokens,
                    temperature=temperature
                )
            
            # Clean generated text
            import re
            text = re.sub(r'\s*<[^>]*>\s*', ' ', output['text']).strip()
            generated_texts.append(text)
            
            # Continue story generation
            text_id = 1
            window_size = min(8, story_length // 2)
            
            while output.get('has_img_output') and len(generated_images) < story_length:
                progress_val = 0.3 + (0.6 * text_id / story_length)
                progress(progress_val, desc=f"🎨 Generating image {text_id}/{story_length-1}...")
                
                # Generate image from text
                if 'img_gen_feat' in output and output['img_gen_feat'] is not None:
                    images_gen = self.models['adapter'].generate(
                        image_embeds=output['img_gen_feat'],
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale
                    )
                    
                    if images_gen:
                        generated_image = images_gen[0]
                        
                        # Add subtitle
                        image_with_subtitle = self.add_subtitle_to_image(generated_image, text)
                        generated_images.append(image_with_subtitle)
                        
                        # Save image
                        image_path = output_dir / f"{text_id:02d}_generated.jpg"
                        image_with_subtitle.save(image_path)
                        
                        # Update image embeddings
                        image_tensor = self.models['image_transform'](generated_image).unsqueeze(0).to(self.device, dtype=self.dtype)
                        with torch.no_grad():
                            new_image_embeds = self.models['visual_encoder'](image_tensor)
                        image_embeds = torch.cat((image_embeds, new_image_embeds), dim=0)
                
                if text_id >= story_length - 1:
                    break
                
                progress_val = 0.3 + (0.6 * (text_id + 0.5) / story_length)
                progress(progress_val, desc=f"📝 Generating text {text_id+1}/{story_length-1}...")
                
                # Prepare next prompt
                prompt = prompt + text + image_tokens
                text_id += 1
                
                # Apply sliding window
                input_ids = self.models['tokenizer'].encode(prompt, add_special_tokens=False)
                while image_embeds.shape[0] > window_size:
                    eoi_prompt_idx = prompt.find(EOI_TOKEN)
                    if eoi_prompt_idx != -1:
                        prompt = prompt[eoi_prompt_idx + len(EOI_TOKEN) + len('[INST]'):]
                        image_embeds = image_embeds[1:]
                        input_ids = self.models['tokenizer'].encode(prompt, add_special_tokens=False)
                    else:
                        break
                
                input_ids = [self.models['tokenizer'].bos_token_id] + input_ids
                
                # Find token positions
                boi_indices = [i for i, token_id in enumerate(input_ids) if token_id == self.boi_token_id]
                eoi_indices = [i for i, token_id in enumerate(input_ids) if token_id == self.eoi_token_id]
                
                input_ids = torch.tensor(input_ids).to(self.device, dtype=torch.long).unsqueeze(0)
                
                ids_cmp_mask = torch.zeros_like(input_ids, dtype=torch.bool)
                for i in range(image_embeds.shape[0]):
                    if i < len(boi_indices) and i < len(eoi_indices):
                        ids_cmp_mask[0, boi_indices[i] + 1:eoi_indices[i]] = True
                
                embeds_cmp_mask = torch.tensor([True] * image_embeds.shape[0]).to(self.device, dtype=torch.bool)
                
                # Generate next text
                with torch.no_grad():
                    output = self.models['agent'].generate(
                        tokenizer=self.models['tokenizer'],
                        input_ids=input_ids,
                        image_embeds=image_embeds,
                        embeds_cmp_mask=embeds_cmp_mask,
                        ids_cmp_mask=ids_cmp_mask,
                        max_new_tokens=max_new_tokens,
                        num_img_gen_tokens=num_img_out_tokens,
                        temperature=temperature
                    )
                
                text = re.sub(r'\s*<[^>]*>\s*', ' ', output['text']).strip()
                generated_texts.append(text)
            
            progress(0.95, desc="💾 Saving story...")
            
            # Save story text
            story_text_path = output_dir / "story_text.txt"
            with open(story_text_path, 'w', encoding='utf-8') as f:
                for i, text in enumerate(generated_texts):
                    f.write(f"Segment {i}: {text}\n\n")
            
            progress(1.0, desc="✅ Story generation complete!")
            
            status_message = f"✅ Generated {len(generated_images)} images and {len(generated_texts)} text segments\n"
            status_message += f"📁 Saved to: {output_dir}"
            
            return generated_images, generated_texts, status_message
            
        except Exception as e:
            logger.error(f"Error generating story: {e}")
            logger.error(traceback.format_exc())
            return [], [f"Error: {str(e)}"], f"❌ Story generation failed: {str(e)}"
    
    def get_system_info(self) -> str:
        """Get system information"""
        try:
            gpu_info = "No GPU available"
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                gpu_info = f"{gpu_name} ({gpu_memory:.1f}GB)"
            
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            info = f"""🖥️ **System Information**
**GPU:** {gpu_info}
**CPU Usage:** {cpu_percent:.1f}%
**Memory:** {memory.percent:.1f}% used ({memory.used/1024**3:.1f}GB / {memory.total/1024**3:.1f}GB)
**Models Loaded:** {'✅ Yes' if self.model_loaded else '❌ No'}
**Device:** {self.device}
**Precision:** {self.dtype}
"""
            return info
            
        except Exception as e:
            return f"Error getting system info: {e}"

def create_interface():
    """Create and configure the Gradio interface"""
    
    # Initialize the SEED-Story interface
    seed_story = SEEDStoryInterface()
    
    # Custom CSS
    css = """
    .gradio-container {
        max-width: 1200px !important;
    }
    .generate-btn {
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4) !important;
        border: none !important;
        color: white !important;
        font-weight: bold !important;
    }
    .status-box {
        background-color: #f0f0f0;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    .image-gallery {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 10px;
    }
    """
    
    with gr.Blocks(
        css=css,
        title="SEED-Story: Multimodal Story Generation",
        theme=gr.themes.Soft()
    ) as interface:
        
        # Header
        gr.Markdown("""
        # 🎬 SEED-Story: Multimodal Story Generation
        
        Generate rich, coherent multimodal stories with consistent characters and style.
        Upload an image and provide opening text to create an engaging visual narrative.
        """)
        
        # System status
        with gr.Row():
            with gr.Column():
                system_info = gr.Markdown(
                    seed_story.get_system_info(),
                    elem_classes=["status-box"]
                )
                refresh_btn = gr.Button("🔄 Refresh System Info", size="sm")
                refresh_btn.click(seed_story.get_system_info, outputs=[system_info])
        
        # Main interface
        with gr.Row():
            # Input column
            with gr.Column(scale=1):
                gr.Markdown("### 📥 Story Input")
                
                input_image = gr.Image(
                    label="Starting Image",
                    type="pil",
                    height=300
                )
                
                initial_text = gr.Textbox(
                    label="Opening Text",
                    placeholder="Once upon a time, in a magical forest...",
                    lines=3,
                    value="Once upon a time..."
                )
                
                gr.Markdown("### ⚙️ Generation Settings")
                
                with gr.Group():
                    story_length = gr.Slider(
                        minimum=3,
                        maximum=25,
                        value=10,
                        step=1,
                        label="Story Length (number of segments)"
                    )
                    
                    max_new_tokens = gr.Slider(
                        minimum=100,
                        maximum=1000,
                        value=500,
                        step=50,
                        label="Max Tokens per Segment"
                    )
                    
                    num_inference_steps = gr.Slider(
                        minimum=20,
                        maximum=100,
                        value=50,
                        step=10,
                        label="Image Generation Steps"
                    )
                    
                    temperature = gr.Slider(
                        minimum=0.1,
                        maximum=1.5,
                        value=0.7,
                        step=0.1,
                        label="Text Generation Temperature"
                    )
                    
                    guidance_scale = gr.Slider(
                        minimum=1.0,
                        maximum=15.0,
                        value=7.5,
                        step=0.5,
                        label="Image Guidance Scale"
                    )
                
                generate_btn = gr.Button(
                    "🎬 Generate Story",
                    variant="primary",
                    size="lg",
                    elem_classes=["generate-btn"]
                )
            
            # Output column
            with gr.Column(scale=2):
                gr.Markdown("### 📖 Generated Story")
                
                status_text = gr.Markdown("Ready to generate stories!")
                
                with gr.Tabs():
                    with gr.TabItem("🖼️ Story Images"):
                        output_gallery = gr.Gallery(
                            label="Generated Story Images",
                            show_label=False,
                            elem_id="gallery",
                            columns=2,
                            rows=3,
                            height="auto"
                        )
                    
                    with gr.TabItem("📝 Story Text"):
                        output_text = gr.Textbox(
                            label="Generated Story Text",
                            lines=15,
                            max_lines=20,
                            show_copy_button=True
                        )
                    
                    with gr.TabItem("📊 Generation Log"):
                        generation_log = gr.Textbox(
                            label="Generation Details",
                            lines=10,
                            interactive=False
                        )
        
        # Example inputs
        with gr.Row():
            gr.Markdown("### 🎯 Quick Start Examples")
            
            examples = gr.Examples(
                examples=[
                    [
                        "examples/fantasy_castle.jpg",
                        "In a mystical realm where magic flows through ancient stones, a young wizard discovers a hidden castle."
                    ],
                    [
                        "examples/space_station.jpg", 
                        "In the year 2087, aboard the space station Aurora, Captain Maya receives a mysterious signal from deep space."
                    ],
                    [
                        "examples/forest_path.jpg",
                        "Sarah stepped into the enchanted forest, not knowing that this path would change her life forever."
                    ]
                ],
                inputs=[input_image, initial_text],
                label="Click to load example"
            )
        
        # Generation function
        def generate_story_wrapper(*args):
            try:
                images, texts, status = seed_story.generate_story(*args)
                
                # Format text output
                formatted_text = ""
                for i, text in enumerate(texts):
                    formatted_text += f"**Segment {i+1}:**\n{text}\n\n"
                
                # Create generation log
                log = f"✅ Story Generation Complete\n"
                log += f"📊 Generated {len(images)} images and {len(texts)} text segments\n"
                log += f"⚙️ Settings: Length={args[2]}, Tokens={args[3]}, Steps={args[4]}\n"
                log += f"🎯 Temperature={args[5]}, Guidance={args[6]}"
                
                return images, formatted_text, status, log
                
            except Exception as e:
                error_msg = f"❌ Error: {str(e)}"
                return [], "", error_msg, f"Error details:\n{traceback.format_exc()}"
        
        # Connect generate button
        generate_btn.click(
            fn=generate_story_wrapper,
            inputs=[
                input_image,
                initial_text,
                story_length,
                max_new_tokens,
                num_inference_steps,
                temperature,
                guidance_scale
            ],
            outputs=[
                output_gallery,
                output_text,
                status_text,
                generation_log
            ],
            show_progress=True
        )
        
        # Footer
        gr.Markdown("""
        ---
        **SEED-Story** by TencentARC - Multimodal Long Story Generation with Large Language Model
        
        📚 [Paper](https://arxiv.org/abs/2407.08683) • 🔗 [GitHub](https://github.com/TencentARC/SEED-Story) • 🤗 [Models](https://huggingface.co/TencentARC/SEED-Story)
        """)
    
    return interface

def main():
    """Main function to launch the interface"""
    try:
        # Create interface
        interface = create_interface()
        
        # Launch configuration
        server_name = os.getenv("GRADIO_SERVER_NAME", "0.0.0.0")
        server_port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
        
        print("🎬 Launching SEED-Story Gradio Interface...")
        print(f"   Server: {server_name}:{server_port}")
        print(f"   Access URL: http://localhost:{server_port}")
        
        # Launch interface
        interface.launch(
            server_name=server_name,
            server_port=server_port,
            share=False,
            show_error=True,
            show_tips=True,
            enable_queue=True,
            max_threads=4
        )
        
    except Exception as e:
        logger.error(f"Failed to launch interface: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()
