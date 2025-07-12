#!/usr/bin/env python3
"""
Simple Comic Generator for SEED-Story
Generates comic panels with text overlays as a fallback when models aren't available
"""

import os
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

import gradio as gr
import torch
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import random
import numpy as np

print("🎨 Starting Simple Comic Generator...")

# Global variable for diffusion pipeline
diffusion_pipe = None

def create_comic_panel(text, panel_num, width=512, height=512):
    """Create a comic-style panel with text"""
    # Create base image with comic book colors
    colors = [
        (255, 230, 200),  # Light peach
        (200, 230, 255),  # Light blue
        (255, 200, 230),  # Light pink
        (230, 255, 200),  # Light green
        (255, 255, 200),  # Light yellow
    ]
    
    bg_color = colors[panel_num % len(colors)]
    img = Image.new('RGB', (width, height), color=bg_color)
    draw = ImageDraw.Draw(img)
    
    # Draw comic panel border
    border_width = 8
    draw.rectangle(
        [(border_width, border_width), (width-border_width, height-border_width)], 
        outline='black', 
        width=border_width
    )
    
    # Draw inner border
    inner_border = 20
    draw.rectangle(
        [(inner_border, inner_border), (width-inner_border, height-inner_border)], 
        outline='black', 
        width=2
    )
    
    # Add panel number
    try:
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 36)
        text_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
    except:
        title_font = ImageFont.load_default()
        text_font = ImageFont.load_default()
    
    # Panel number in corner
    panel_text = f"#{panel_num}"
    draw.text((30, 30), panel_text, fill='black', font=title_font)
    
    # Add decorative elements
    # Speech bubble
    bubble_x = width // 2
    bubble_y = height // 3
    bubble_w = width - 80
    bubble_h = 150
    
    # Draw speech bubble
    draw.ellipse(
        [(bubble_x - bubble_w//2, bubble_y - bubble_h//2), 
         (bubble_x + bubble_w//2, bubble_y + bubble_h//2)],
        fill='white',
        outline='black',
        width=3
    )
    
    # Bubble tail
    points = [
        (bubble_x - 20, bubble_y + bubble_h//2 - 10),
        (bubble_x - 40, bubble_y + bubble_h//2 + 40),
        (bubble_x + 20, bubble_y + bubble_h//2 - 10)
    ]
    draw.polygon(points, fill='white', outline='black')
    
    # Wrap text for speech bubble
    words = text.split()
    lines = []
    current_line = []
    max_width = bubble_w - 40
    
    for word in words:
        test_line = ' '.join(current_line + [word])
        bbox = draw.textbbox((0, 0), test_line, font=text_font)
        if bbox[2] - bbox[0] <= max_width:
            current_line.append(word)
        else:
            if current_line:
                lines.append(' '.join(current_line))
            current_line = [word]
    
    if current_line:
        lines.append(' '.join(current_line))
    
    # Draw text in bubble
    text_y = bubble_y - (len(lines) * 12)
    for line in lines[:4]:  # Max 4 lines
        bbox = draw.textbbox((0, 0), line, font=text_font)
        text_x = bubble_x - (bbox[2] - bbox[0]) // 2
        draw.text((text_x, text_y), line, fill='black', font=text_font)
        text_y += 24
    
    # Add comic effects
    effects = ['POW!', 'ZAP!', 'BOOM!', 'WHOOSH!', 'BAM!']
    effect = random.choice(effects)
    effect_x = random.randint(50, width-150)
    effect_y = random.randint(height//2, height-100)
    
    # Draw effect background
    draw.ellipse(
        [(effect_x-40, effect_y-20), (effect_x+40, effect_y+20)],
        fill='yellow',
        outline='red',
        width=3
    )
    draw.text((effect_x-30, effect_y-10), effect, fill='red', font=title_font)
    
    return img

def try_load_diffusion_model():
    """Try to load Stable Diffusion model if available"""
    global diffusion_pipe
    
    try:
        from diffusers import DiffusionPipeline
        
        models_dir = Path(os.environ.get('SEED_STORY_MODELS_DIR', '/app/pretrained'))
        sdxl_path = models_dir / "stable-diffusion-xl-base-1.0"
        
        if sdxl_path.exists():
            print("🔄 Loading Stable Diffusion XL...")
            diffusion_pipe = DiffusionPipeline.from_pretrained(
                str(sdxl_path),
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                use_safetensors=True,
                variant="fp16" if torch.cuda.is_available() else None
            )
            
            if torch.cuda.is_available():
                diffusion_pipe = diffusion_pipe.to("cuda")
                diffusion_pipe.enable_attention_slicing()
            
            print("✅ Stable Diffusion XL loaded!")
            return True
    except Exception as e:
        print(f"⚠️  Could not load diffusion model: {e}")
    
    return False

def generate_panel_with_ai(prompt, panel_num):
    """Generate panel using AI if available, otherwise use fallback"""
    global diffusion_pipe
    
    if diffusion_pipe is not None:
        try:
            print(f"🤖 Generating AI image for panel {panel_num}...")
            
            # Comic-style prompt
            enhanced_prompt = f"comic book panel, colorful illustration: {prompt}"
            
            image = diffusion_pipe(
                prompt=enhanced_prompt,
                negative_prompt="blurry, realistic photo, dark",
                num_inference_steps=20,
                guidance_scale=7.5,
                width=512,
                height=512
            ).images[0]
            
            # Add panel number overlay
            draw = ImageDraw.Draw(image)
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 36)
            except:
                font = ImageFont.load_default()
            
            draw.text((20, 20), f"#{panel_num}", fill='white', font=font, stroke_width=2, stroke_fill='black')
            
            return image
            
        except Exception as e:
            print(f"⚠️  AI generation failed: {e}")
    
    # Fallback to comic panel creation
    return create_comic_panel(prompt, panel_num)

def generate_comic_story(prompt, num_panels, use_ai=True):
    """Generate a complete comic story"""
    print(f"📚 Generating {num_panels}-panel comic story: '{prompt}'")
    
    # Try to load AI model if requested and not already loaded
    if use_ai and diffusion_pipe is None:
        try_load_diffusion_model()
    
    # Generate story based on prompt
    story_elements = []
    
    # Parse prompt for story elements
    if "cat" in prompt.lower():
        story_elements = [
            f"A curious cat discovers {prompt}",
            f"The cat investigates further",
            f"An unexpected twist occurs!",
            f"The cat saves the day",
            f"Happy ending with {prompt}"
        ]
    elif "magic" in prompt.lower():
        story_elements = [
            f"A wizard finds {prompt}",
            f"Magic sparkles everywhere",
            f"A spell goes wrong!",
            f"The wizard fixes everything",
            f"Peace returns to the realm"
        ]
    elif "space" in prompt.lower():
        story_elements = [
            f"Astronauts discover {prompt}",
            f"They explore the unknown",
            f"Aliens appear!",
            f"First contact is made",
            f"A new era begins"
        ]
    else:
        # Generic adventure
        story_elements = [
            f"Our hero encounters {prompt}",
            f"The adventure begins",
            f"A challenge appears!",
            f"The hero overcomes it",
            f"Victory is achieved"
        ]
    
    # Adjust to requested number of panels
    while len(story_elements) < num_panels:
        story_elements.append(f"The story of {prompt} continues...")
    story_elements = story_elements[:num_panels]
    
    # Generate images
    images = []
    story_text = f"**Comic Story: {prompt}**\n\n"
    
    for i, element in enumerate(story_elements):
        panel_num = i + 1
        print(f"Creating panel {panel_num}/{num_panels}: {element}")
        
        if use_ai and diffusion_pipe is not None:
            image = generate_panel_with_ai(element, panel_num)
        else:
            image = create_comic_panel(element, panel_num)
        
        images.append(image)
        story_text += f"**Panel {panel_num}:** {element}\n"
    
    # Add system info
    story_text += f"\n**Generation Info:**\n"
    story_text += f"- Mode: {'AI-Enhanced' if (use_ai and diffusion_pipe is not None) else 'Classic Comic Style'}\n"
    story_text += f"- GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}\n"
    story_text += f"- Panels: {len(images)}\n"
    
    return images, story_text

def create_interface():
    """Create Gradio interface"""
    with gr.Blocks(title="Comic Story Generator", theme=gr.themes.Soft()) as interface:
        gr.Markdown("# 🎨 Comic Story Generator")
        gr.Markdown("Create amazing comic stories with AI or classic comic style!")
        
        with gr.Row():
            with gr.Column(scale=1):
                prompt_input = gr.Textbox(
                    label="Story Idea",
                    placeholder="Enter your story idea...",
                    value="a brave cat exploring a magical forest",
                    lines=2
                )
                
                num_panels = gr.Slider(
                    minimum=1,
                    maximum=8,
                    value=4,
                    step=1,
                    label="Number of Panels"
                )
                
                use_ai = gr.Checkbox(
                    label="Use AI Generation (if available)",
                    value=True
                )
                
                generate_btn = gr.Button("🎬 Generate Comic!", variant="primary", size="lg")
                
                gr.Markdown("### 💡 Try these ideas:")
                gr.Markdown("- a cat discovering magic powers\n- space adventure with aliens\n- underwater treasure hunt\n- robot learning to paint")
            
            with gr.Column(scale=2):
                output_gallery = gr.Gallery(
                    label="Comic Panels",
                    show_label=True,
                    elem_id="gallery",
                    columns=2,
                    rows=2,
                    height="auto"
                )
                
                output_text = gr.Textbox(
                    label="Story Description",
                    lines=8
                )
        
        # Connect the button
        generate_btn.click(
            fn=generate_comic_story,
            inputs=[prompt_input, num_panels, use_ai],
            outputs=[output_gallery, output_text]
        )
    
    return interface

if __name__ == "__main__":
    print("🚀 Starting Comic Story Generator...")
    
    # Check GPU
    if torch.cuda.is_available():
        print(f"✅ GPU Available: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️  No GPU detected - using CPU mode")
    
    # Create and launch interface
    interface = create_interface()
    
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        share=False,
        show_api=False,
        allowed_paths=["/app"],
        quiet=False,
        prevent_thread_lock=False
    )