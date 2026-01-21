"""
OmniControlGP - Subject-Driven Image Generation
GPU-optimized version with mmgp memory management

Based on: https://github.com/deepbeepmeep/OminiControlGP
"""

import os
import sys
import argparse
import gradio as gr
import torch
from PIL import Image
from diffusers.pipelines import FluxPipeline
from diffusers import FluxTransformer2DModel
import numpy as np

# Disable analytics
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

# Import from src directory
sys.path.insert(0, '/app')
from src.flux.condition import Condition
from src.flux.generate import seed_everything, generate

# Try to import mmgp for GPU Poor optimization
try:
    from mmgp import offload
    MMGP_AVAILABLE = True
    print("✅ mmgp module loaded - GPU Poor optimization enabled")
except ImportError:
    MMGP_AVAILABLE = False
    print("⚠️  mmgp module not available - running in standard mode")

# Global pipeline variable
pipe = None
use_int8 = False


def get_gpu_memory():
    """Get total GPU memory in GB"""
    try:
        return torch.cuda.get_device_properties(0).total_memory / 1024**3
    except:
        return 0


def init_pipeline(profile_no=3, verbose_level=1):
    """Initialize the FLUX pipeline with OminiControl LoRA"""
    global pipe
    
    print(f"🔧 Initializing pipeline with profile {profile_no}...")
    
    gpu_memory = get_gpu_memory()
    print(f"📊 GPU Memory: {gpu_memory:.2f} GB")
    
    # Determine if we should use int8 quantization
    use_quantization = (profile_no >= 4 or gpu_memory < 16)
    
    # Get HuggingFace token if available
    hf_token = os.environ.get("HF_TOKEN", None)
    
    try:
        # Load pipeline to CPU first (mmgp will handle device management)
        print("📦 Loading FLUX.1-schnell pipeline to CPU...")
        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell",
            torch_dtype=torch.bfloat16,
            cache_dir=os.environ.get("MODEL_CACHE_DIR", "/app/models"),
            token=hf_token
        )
        pipe = pipe.to("cpu")  # Load to CPU - mmgp will manage devices
        
        # Load OminiControl LoRA weights
        print("📥 Loading OminiControl LoRA weights...")
        pipe.load_lora_weights(
            "Yuanshi/OminiControl",
            weight_name="omini/subject_512.safetensors",
            adapter_name="subject",
            cache_dir=os.environ.get("MODEL_CACHE_DIR", "/app/models"),
            token=hf_token
        )
        
        # Apply mmgp offloading - this handles all device management
        if MMGP_AVAILABLE:
            print(f"⚡ Applying mmgp profile {profile_no}...")
            # Get memory reservation percentage (default 0.75 for Profile 3 - worked well before)
            perc_reserved_mem = float(os.environ.get("MMGP_PERC_RESERVED_MEM_MAX", "0.75"))
            print(f"   Reserved RAM: {int(perc_reserved_mem*100)}% (for {'full' if perc_reserved_mem >= 0.70 else 'partial'} pinning)")
            
            # Get budget override - use 0 to keep model in VRAM (no offloading between generations)
            # This prevents the 80-second reload penalty on each generation
            budget_override = os.environ.get("MMGP_BUDGETS", None)
            if budget_override is not None:
                budget_value = int(budget_override)
                print(f"   Budget override: {budget_value} MB (0 = unlimited VRAM, keeps model loaded)")
                offload.profile(
                    pipe,
                    profile_no=int(profile_no),
                    verboseLevel=int(verbose_level),
                    quantizeTransformer=False,
                    perc_reserved_mem_max=perc_reserved_mem,
                    budgets=budget_value
                )
            else:
                offload.profile(
                    pipe,
                    profile_no=int(profile_no),
                    verboseLevel=int(verbose_level),
                    quantizeTransformer=False,
                    perc_reserved_mem_max=perc_reserved_mem
                )
            print("✅ mmgp profiling complete!")
        else:
            print("⚠️  mmgp not available, loading to CUDA...")
            pipe = pipe.to("cuda")
        
        print("✅ Pipeline initialized successfully!")
        
    except Exception as e:
        print(f"❌ Error initializing pipeline: {e}")
        raise


def process_image_and_text(image, text, num_steps=8, seed_value=None):
    """Process input image and text prompt to generate controlled image"""
    global pipe
    
    if pipe is None:
        return None, "❌ Pipeline not initialized. Please wait..."
    
    if image is None:
        return None, "⚠️ Please upload an image"
    
    if not text or text.strip() == "":
        return None, "⚠️ Please enter a text prompt"
    
    try:
        # Center crop and resize image
        w, h = image.size
        min_size = min(w, h)
        image = image.crop(
            (
                (w - min_size) // 2,
                (h - min_size) // 2,
                (w + min_size) // 2,
                (h + min_size) // 2,
            )
        )
        image = image.resize((512, 512))
        
        # Create condition with position delta
        condition = Condition("subject", image, position_delta=(0, 32))
        
        # Set seed if provided
        if seed_value is not None and seed_value >= 0:
            seed_everything(seed_value)
        else:
            seed_everything()
        
        print(f"🎨 Generating image with prompt: {text[:50]}...")
        
        # Generate image - mmgp handles all device management automatically
        result_img = generate(
            pipe,
            prompt=text.strip(),
            conditions=[condition],
            num_inference_steps=int(num_steps),
            height=512,
            width=512,
        ).images[0]
        
        print("✅ Image generated successfully!")
        return result_img, "✅ Generation complete!"
        
    except torch.cuda.OutOfMemoryError:
        return None, "❌ GPU out of memory! Try:\n- Using dockerfile-gpu-poor\n- Reducing inference steps\n- Restarting the container"
    except Exception as e:
        error_msg = f"❌ Generation failed: {str(e)}"
        print(error_msg)
        return None, error_msg


def get_samples():
    """Get example images and prompts"""
    sample_list = [
        {
            "image": "assets/penguin.jpg",
            "text": "On Christmas evening, on a crowded sidewalk, this item sits on the road, covered in snow and wearing a Christmas hat, holding a sign that reads 'Omini Control!'",
        },
        {
            "image": "assets/oranges.jpg",
            "text": "A very close up view of this item. It is placed on a wooden table. The background is a dark room, the TV is on, and the screen is showing a cooking show. With text on the screen that reads 'Omini Control!'",
        },
        {
            "image": "assets/rc_car.jpg",
            "text": "A film style shot. On the moon, this item drives across the moon surface. The background is that Earth looms large in the foreground.",
        },
    ]
    
    examples = []
    for sample in sample_list:
        try:
            if os.path.exists(sample["image"]):
                examples.append([Image.open(sample["image"]), sample["text"], 8, -1])
        except:
            pass
    
    return examples if examples else None


def create_interface():
    """Create Gradio interface"""
    with gr.Blocks(title="OminiControlGP") as demo:
        gr.Markdown(
            """
            # 🎨 OminiControlGP - Subject-Driven Image Generation
            ### Powered by FLUX.1-schnell + OminiControl LoRA
            
            Upload an object image and describe the scene you want. The model will generate a new image
            with your object in the described context. Use phrases like "this item", "the object", or "it"
            to refer to the subject.
            
            **Tips:**
            - Works best with objects (not people)
            - Images are auto-cropped to 512x512
            - 8 inference steps recommended for speed
            - Use descriptive prompts with context
            """
        )
        
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(
                    type="pil",
                    label="📸 Input Image",
                    height=400
                )
                input_text = gr.Textbox(
                    lines=3,
                    label="💬 Text Prompt",
                    placeholder="Describe the scene... (e.g., 'In a bright room, this item is on a wooden table')"
                )
                
                with gr.Row():
                    num_steps = gr.Slider(
                        minimum=4,
                        maximum=20,
                        value=8,
                        step=1,
                        label="🔄 Inference Steps"
                    )
                    seed = gr.Number(
                        value=-1,
                        label="🎲 Seed (-1 for random)",
                        precision=0
                    )
                
                generate_btn = gr.Button("🚀 Generate Image", variant="primary")
                
            with gr.Column():
                output_image = gr.Image(
                    label="🖼️ Generated Image",
                    height=400
                )
                status_text = gr.Textbox(
                    label="📊 Status",
                    interactive=False
                )
        
        # System info
        with gr.Accordion("📊 System Information", open=False):
            gpu_info = f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU'}"
            vram_info = f"VRAM: {get_gpu_memory():.2f} GB"
            profile_info = f"Profile: {os.environ.get('OMNI_PROFILE', 'N/A')}"
            mmgp_info = f"mmgp: {'Enabled' if MMGP_AVAILABLE else 'Disabled'}"
            
            gr.Markdown(f"""
            - {gpu_info}
            - {vram_info}
            - {profile_info}
            - {mmgp_info}
            
            **Memory Profiles:**
            - Profile 1: 16+ GB VRAM (Fastest)
            - Profile 3: 12-16 GB VRAM (Default)
            - Profile 4: 8-12 GB VRAM (Low VRAM)
            - Profile 5: 6-8 GB VRAM (GPU Poor)
            """)
        
        # Guidelines
        with gr.Accordion("💡 Usage Guidelines", open=False):
            gr.Markdown("""
            **Best Practices:**
            1. Use clear, descriptive prompts
            2. Reference the object as "this item", "the object", or "it"
            3. Include environmental context (location, lighting, background)
            4. Works best with objects (toys, tools, furniture, etc.)
            5. Images are automatically center-cropped and resized
            
            **Example Prompts:**
            - "On a beach at sunset, this item sits on the sand with waves in the background"
            - "In a modern kitchen, this item is placed on a marble countertop"
            - "This object floating in space with Earth visible in the distance"
            """)
        
        # Examples
        examples = get_samples()
        if examples:
            gr.Examples(
                examples=examples,
                inputs=[input_image, input_text, num_steps, seed],
                outputs=[output_image, status_text],
                fn=process_image_and_text,
                cache_examples=False,
                label="📚 Example Images & Prompts"
            )
        
        # Connect the button
        generate_btn.click(
            fn=process_image_and_text,
            inputs=[input_image, input_text, num_steps, seed],
            outputs=[output_image, status_text]
        )
    
    return demo


def main():
    parser = argparse.ArgumentParser(description="OminiControlGP Gradio Interface")
    parser.add_argument(
        '--profile',
        type=int,
        default=int(os.environ.get('OMNI_PROFILE', '3')),
        help='Memory profile (1-5, lower=more VRAM required)'
    )
    parser.add_argument(
        '--verbose',
        type=int,
        default=int(os.environ.get('OMNI_VERBOSE', '1')),
        help='Verbosity level for mmgp (0-2)'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=int(os.environ.get('GRADIO_SERVER_PORT', 7860)),
        help='Port to run Gradio server'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🚀 OminiControlGP Starting...")
    print("=" * 60)
    
    # Initialize pipeline
    try:
        init_pipeline(profile_no=args.profile, verbose_level=args.verbose)
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        sys.exit(1)
    
    # Create and launch interface
    print("\n🌐 Creating Gradio interface...")
    demo = create_interface()
    
    print(f"\n✅ Launching on port {args.port}...")
    print("=" * 60)
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=False,
        debug=False
    )


if __name__ == "__main__":
    main()
