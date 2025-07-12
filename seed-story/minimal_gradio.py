#!/usr/bin/env python3
"""
Minimal SEED-Story Interface with Image Generation
Simple story generation with comic panel images
"""

import os
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

import gradio as gr
import torch
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import random

print("🔧 Starting SEED-Story interface with image generation...")

def check_models():
    """Check if models are available"""
    models_dir = Path(os.environ.get('SEED_STORY_MODELS_DIR', '/app/pretrained'))
    print(f"🟢 STEP 13: Models directory: {models_dir}")
    
    models_status = {}
    required_models = [
        "stable-diffusion-xl-base-1.0",
        "Llama-2-7b-hf", 
        "Qwen-VL-Chat"
    ]
    print(f"🟢 STEP 14: Checking {len(required_models)} models...")
    
    for i, model in enumerate(required_models):
        print(f"🟢 STEP 15.{i+1}: Checking model {model}...")
        model_path = models_dir / model
        models_status[model] = "✅ Available" if model_path.exists() else "❌ Missing"
        print(f"🟢 STEP 15.{i+1}b: {model} = {models_status[model]}")
    
    print("🟢 STEP 16: ✅ check_models completed")
    return models_status

def load_diffusion_model():
    """Load Stable Diffusion XL for image generation"""
    try:
        print("🟢 Loading Stable Diffusion XL...")
        from diffusers import StableDiffusionXLPipeline
        
        models_dir = Path(os.environ.get('SEED_STORY_MODELS_DIR', '/app/pretrained'))
        sdxl_path = models_dir / "stable-diffusion-xl-base-1.0"
        
        if sdxl_path.exists():
            pipe = StableDiffusionXLPipeline.from_pretrained(
                str(sdxl_path),
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                use_safetensors=True,
                variant="fp16" if torch.cuda.is_available() else None
            )
            
            if torch.cuda.is_available():
                pipe = pipe.to("cuda")
            
            print("✅ Stable Diffusion XL loaded successfully!")
            return pipe
        else:
            print("❌ SDXL model not found")
            return None
    except Exception as e:
        print(f"❌ Error loading SDXL: {e}")
        return None

# Global variable to store the diffusion pipeline
diffusion_pipe = None

def create_placeholder_image(text, panel_num):
    """Create a placeholder comic panel image with text"""
    try:
        # Create a comic-style image
        img_width, img_height = 512, 512
        img = Image.new('RGB', (img_width, img_height), color='lightblue')
        
        draw = ImageDraw.Draw(img)
        
        # Try to load a font, fall back to default if not available
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
            small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        except:
            font = ImageFont.load_default()
            small_font = ImageFont.load_default()
        
        # Draw panel border
        border_color = 'black'
        draw.rectangle([(10, 10), (img_width-10, img_height-10)], outline=border_color, width=3)
        
        # Draw panel number
        draw.text((20, 20), f"Panel {panel_num}", fill='black', font=font)
        
        # Wrap and draw text
        max_width = img_width - 40
        words = text.split()
        lines = []
        current_line = []
        
        for word in words:
            test_line = ' '.join(current_line + [word])
            bbox = draw.textbbox((0, 0), test_line, font=small_font)
            if bbox[2] - bbox[0] <= max_width:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]
        
        if current_line:
            lines.append(' '.join(current_line))
        
        # Draw text lines
        y_start = img_height // 2 - (len(lines) * 20) // 2
        for i, line in enumerate(lines[:8]):  # Limit to 8 lines
            draw.text((20, y_start + i * 20), line, fill='black', font=small_font)
        
        return img
        
    except Exception as e:
        print(f"Error creating placeholder image: {e}")
        # Return a simple colored rectangle as fallback
        img = Image.new('RGB', (512, 512), color=f'hsl({(panel_num * 50) % 360}, 70%, 80%)')
        return img

def generate_image_with_diffusion(prompt, panel_num):
    """Generate an image using Stable Diffusion XL"""
    global diffusion_pipe
    
    try:
        if diffusion_pipe is None:
            print("🔄 Loading diffusion model...")
            diffusion_pipe = load_diffusion_model()
        
        if diffusion_pipe is not None:
            print(f"🎨 Generating image for panel {panel_num}...")
            
            # Enhance prompt for comic-style generation
            enhanced_prompt = f"comic book style, colorful illustration, {prompt}, detailed artwork, vibrant colors, comic panel"
            
            # Generate image
            image = diffusion_pipe(
                prompt=enhanced_prompt,
                negative_prompt="blurry, low quality, distorted, dark, gloomy",
                num_inference_steps=30,  # Reduced for speed
                guidance_scale=7.5,
                width=512,
                height=512
            ).images[0]
            
            print(f"✅ Generated image for panel {panel_num}")
            return image
        else:
            print("⚠️ Diffusion model not available, using placeholder")
            return create_placeholder_image(prompt, panel_num)
            
    except Exception as e:
        print(f"❌ Error generating image: {e}")
        return create_placeholder_image(prompt, panel_num)

def generate_story(prompt, num_panels):
    """Generate a comic story with images based on the prompt"""
    print(f"🟢 STEP 20: generate_story called with prompt='{prompt}', panels={num_panels}")
    
    # Check if models are available
    models_dir = Path(os.environ.get('SEED_STORY_MODELS_DIR', '/app/pretrained'))
    print(f"🟢 STEP 20a: Models directory: {models_dir}")
    
    try:
        print("🎬 STEP 20b: Starting story generation with images...")
        
        # Story templates based on prompt themes
        story_templates = {
            "cat": [
                "A curious cat named Whiskers discovers a mysterious glowing orb in the garden.",
                "The orb transports Whiskers to a magical realm filled with floating fish and yarn balls.",
                "Whiskers meets the Cat Queen who explains that the realm is in danger from evil dogs.",
                "Armed with magical whiskers, Whiskers must save the realm and return home.",
                "Whiskers successfully defeats the evil dogs and becomes the hero of the cat realm."
            ],
            "magic": [
                "A young wizard discovers an ancient spellbook hidden in their grandmother's attic.",
                "The first spell accidentally turns their pet hamster into a dragon.",
                "Together, they must journey to the Enchanted Forest to find the reversal spell.",
                "They encounter mystical creatures who help them learn the true power of friendship.",
                "With newfound wisdom, they master the magic and restore balance to their world."
            ],
            "space": [
                "Captain Nova receives a distress signal from a distant galaxy.",
                "Their spaceship travels through a wormhole to reach the alien planet.",
                "They discover an ancient civilization fighting against robotic invaders.",
                "Nova helps the aliens build a powerful defense system using future technology.",
                "The planet is saved, and Nova is honored as an intergalactic hero."
            ],
            "adventure": [
                "Explorer Maya finds a treasure map in an old library book.",
                "The map leads her through dangerous jungles and across raging rivers.",
                "She discovers a hidden temple guarded by ancient puzzles and traps.",
                "Inside the temple, Maya finds not gold, but a cure for a rare disease.",
                "Maya returns home and shares the cure, saving countless lives."
            ]
        }
        
        # Determine story theme
        theme = "adventure"  # default
        prompt_lower = prompt.lower()
        for key in story_templates.keys():
            if key in prompt_lower:
                theme = key
                break
        
        print(f"🟢 STEP 20c: Using theme '{theme}' for story generation")
        
        # Generate story panels
        story_panels = story_templates[theme][:num_panels]
        
        # If we need more panels, extend with creative variations
        while len(story_panels) < num_panels:
            additional_panel = f"The adventure continues as our hero faces new challenges related to {prompt}..."
            story_panels.append(additional_panel)
        
        # Generate images for each panel
        generated_images = []
        story_text = f"🎬 **Generated {num_panels}-Panel Comic Story:**\n"
        story_text += f"**Theme:** {theme.title()} Adventure\n\n"
        
        for i, panel_text in enumerate(story_panels):
            print(f"🟢 STEP 20d.{i+1}: Generating panel {i+1}/{num_panels}")
            
            # Create image prompt based on panel text
            image_prompt = f"{panel_text.split('.')[0]}. {prompt}."
            
            # Generate image for this panel
            panel_image = generate_image_with_diffusion(image_prompt, i+1)
            generated_images.append(panel_image)
            
            # Add text to story
            story_text += f"**Panel {i+1}:**\n{panel_text}\n\n"
            
            print(f"✅ Panel {i+1} completed")
        
        # Add system info
        story_text += f"\n**System Info:**\n"
        story_text += f"- CUDA Available: {torch.cuda.is_available()}\n"
        story_text += f"- GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}\n"
        story_text += f"- Models Directory: {models_dir}\n"
        story_text += f"- Generated {len(generated_images)} comic panels\n"
        
        print("🟢 STEP 21: ✅ generate_story completed")
        return generated_images, story_text
        
    except Exception as e:
        print(f"🔴 STEP 20x: Error in story generation: {e}")
        import traceback
        traceback.print_exc()
        
        # Create fallback images
        fallback_images = []
        for i in range(num_panels):
            fallback_img = create_placeholder_image(f"Panel {i+1}: A story about {prompt}", i+1)
            fallback_images.append(fallback_img)
        
        story_text = f"⚠️ **Fallback Comic Story (Based on: '{prompt}'):**\n\n"
        
        for i in range(num_panels):
            story_text += f"**Panel {i+1}:** A wonderful story about {prompt} unfolds in this panel...\n\n"
        
        story_text += f"\n**Error:** {str(e)}\n"
        story_text += f"**System Info:**\n"
        story_text += f"- CUDA Available: {torch.cuda.is_available()}\n"
        story_text += f"- GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}\n"
        story_text += f"- Models Directory: {models_dir}\n"
        
        return fallback_images, story_text


def create_interface():
    """Create the SEED-Story interface with image generation"""
    print("🟢 STEP 17: Creating SEED-Story interface with comic generation...")
    
    # Check models first
    models_status = check_models()
    
    # Create interface with comic generation capabilities
    with gr.Blocks(title="SEED-Story Comic Generator") as interface:
        gr.Markdown("# 🎬 SEED-Story Comic Generator")
        gr.Markdown("Generate comic stories with images!")
        
        # Model status display
        status_text = "**Model Status:**\n"
        for model, status in models_status.items():
            status_text += f"- {model}: {status}\n"
        
        gr.Markdown(status_text)
        
        with gr.Row():
            with gr.Column(scale=1):
                prompt_input = gr.Textbox(
                    label="Story Prompt",
                    placeholder="Enter your story idea (e.g., 'a cat exploring space', 'magic adventure', etc.)...",
                    lines=3
                )
                num_panels = gr.Slider(
                    minimum=1,
                    maximum=8,
                    value=4,
                    step=1,
                    label="Number of Comic Panels"
                )
                generate_btn = gr.Button("🎨 Generate Comic Story", variant="primary", size="lg")
            
            with gr.Column(scale=2):
                gr.Markdown("### 🖼️ Generated Comic Panels")
                output_gallery = gr.Gallery(
                    label="Comic Story Images",
                    show_label=False,
                    elem_id="gallery",
                    columns=2,
                    rows=2,
                    height="auto",
                    allow_preview=True
                )
                
                output_text = gr.Textbox(
                    label="Story Text",
                    lines=10,
                    max_lines=15
                )
        
        # Set up the generate button to return both images and text
        generate_btn.click(
            fn=generate_story,
            inputs=[prompt_input, num_panels],
            outputs=[output_gallery, output_text],
            api_name=False  # Disable API to avoid schema issues
        )
        
        # Add some example prompts
        gr.Markdown("### 💡 Example Prompts")
        gr.Markdown("- `a cat eating food and discovering magic`")
        gr.Markdown("- `space explorer discovering alien planets`") 
        gr.Markdown("- `wizard learning spells in magic school`")
        gr.Markdown("- `adventure through mysterious forest`")
        
        print("🟢 STEP 18: Interface created successfully")
    
    return interface

if __name__ == "__main__":
    print("🚀 Starting SEED-Story Comic Generator...")
    
    try:
        interface = create_interface()
        print("🌐 Launching interface...")
        # Launch with simple configuration to avoid compatibility issues
        interface.launch(
            server_name="0.0.0.0",
            server_port=7860,
            show_error=True,
            share=False,  # Disable sharing for Docker (use port mapping instead)
            inbrowser=False,
            enable_queue=False,  # Disable queue to avoid schema issues
            show_api=False  # Disable API documentation to avoid schema bug
        )
        
    except Exception as e:
        print(f"❌ Failed to start interface: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)