#!/usr/bin/env python3
"""
Generate application icons for Deep-Live-Cam
Creates ICO for Windows and PNG for Linux from a base design
"""
import os
from PIL import Image, ImageDraw, ImageFont

# Get script directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MEDIA_DIR = os.path.join(SCRIPT_DIR, 'media')


def create_icon():
    """Create a simple app icon"""
    
    # Create base image (512x512 for high quality)
    size = 512
    img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Background - dark gradient circle (simulated with layers)
    # Outer circle - dark blue
    draw.ellipse([0, 0, size-1, size-1], fill=(25, 25, 50, 255))
    
    # Inner circle - slightly lighter
    margin = 30
    draw.ellipse([margin, margin, size-margin-1, size-margin-1], fill=(40, 40, 80, 255))
    
    # Draw "DLC" text
    try:
        font = ImageFont.truetype("arial.ttf", 180)
    except:
        font = ImageFont.load_default()
    
    # Center the text
    text = "DLC"
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x = (size - text_width) // 2
    y = (size - text_height) // 2 - 10
    
    # Draw text with glow effect
    # Outer glow
    draw.text((x-3, y-3), text, fill=(100, 150, 255, 200), font=font)
    draw.text((x+3, y+3), text, fill=(100, 150, 255, 200), font=font)
    # Main text
    draw.text((x, y), text, fill=(180, 210, 255, 255), font=font)
    
    return img


def save_icons(img):
    """Save icon in various formats"""
    
    # Windows ICO (multi-resolution)
    ico_sizes = [(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
    
    # Save as ICO using PIL's built-in ICO save
    ico_path = os.path.join(MEDIA_DIR, 'icon.ico')
    
    # Create list of images at different sizes
    icons = []
    for size in ico_sizes:
        icons.append(img.resize(size, Image.Resampling.LANCZOS))
    
    # Save as ICO - use the 256x256 as base
    img.resize((256, 256), Image.Resampling.LANCZOS).save(ico_path, format='ICO')
    print(f"Created: {ico_path}")
    
    # Also save as PNG for Linux
    png_path = os.path.join(MEDIA_DIR, 'icon.png')
    img.resize((256, 256), Image.Resampling.LANCZOS).save(png_path, 'PNG')
    print(f"Created: {png_path}")
    
    return ico_path, png_path


if __name__ == '__main__':
    print("Generating Deep-Live-Cam app icons...")
    img = create_icon()
    save_icons(img)
    print("Icon generation complete!")