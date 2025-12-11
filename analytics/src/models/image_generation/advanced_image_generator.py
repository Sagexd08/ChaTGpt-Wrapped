"""
Advanced Image Generation Model
Custom diffusion model and GAN integration for cover art generation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import List, Tuple, Dict, Any
import io

class CustomConvBlock(nn.Module):
    """Custom convolutional block for generator"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, 
            stride=stride, padding=kernel_size // 2, bias=False
        )
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.activation(self.norm(self.conv(x)))

class CustomUpBlock(nn.Module):
    """Upsampling block for image generation"""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = CustomConvBlock(in_channels, out_channels)
        self.skip_conv = CustomConvBlock(in_channels + out_channels, out_channels)
    
    def forward(self, x, skip=None):
        x = self.up(x)
        x = self.conv(x)
        
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.skip_conv(x)
        
        return x

class CustomGANGenerator(nn.Module):
    """Custom GAN generator for cyberpunk neon aesthetics"""
    
    def __init__(self, latent_dim: int = 256, img_size: int = 512):
        super().__init__()
        self.latent_dim = latent_dim
        self.img_size = img_size
        
        # Linear projection
        self.fc = nn.Linear(latent_dim, 512 * 8 * 8)
        
        # Upsampling layers
        self.layers = nn.Sequential(
            CustomUpBlock(512, 512),  # 8x8 -> 16x16
            CustomUpBlock(512, 256),  # 16x16 -> 32x32
            CustomUpBlock(256, 128),  # 32x32 -> 64x64
            CustomUpBlock(128, 64),   # 64x64 -> 128x128
            CustomUpBlock(64, 32),    # 128x128 -> 256x256
            CustomUpBlock(32, 16),    # 256x256 -> 512x512
        )
        
        # Final convolution
        self.final_conv = nn.Conv2d(16, 3, kernel_size=3, padding=1)
        self.final_activation = nn.Tanh()
    
    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 512, 8, 8)
        x = self.layers(x)
        x = self.final_conv(x)
        return self.final_activation(x)

class CustomGANDiscriminator(nn.Module):
    """Custom GAN discriminator"""
    
    def __init__(self, img_size: int = 512):
        super().__init__()
        
        self.layers = nn.Sequential(
            CustomConvBlock(3, 16, stride=2),
            CustomConvBlock(16, 32, stride=2),
            CustomConvBlock(32, 64, stride=2),
            CustomConvBlock(64, 128, stride=2),
            CustomConvBlock(128, 256, stride=2),
            CustomConvBlock(256, 512, stride=2),
        )
        
        self.fc = nn.Sequential(
            nn.Linear(512 * 8 * 8, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.layers(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class AdvancedImageGenerator:
    """Advanced image generation using multiple techniques"""
    
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        
        # Initialize Stable Diffusion
        self.pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        )
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.to(device)
        self.pipe.enable_attention_slicing()
        
        # Custom GAN
        self.gan_generator = CustomGANGenerator().to(device)
        self.gan_discriminator = CustomGANDiscriminator().to(device)
        
        # Neon color palette
        self.neon_colors = {
            'pink': (255, 0, 110),
            'blue': (0, 217, 255),
            'green': (57, 255, 20),
            'purple': (177, 0, 255),
            'cyan': (0, 255, 255),
            'magenta': (255, 0, 255),
        }
    
    def generate_cover_art(self,
                          title: str,
                          subtitle: str,
                          persona: str,
                          primary_emotion: str = 'positive',
                          custom_prompt: str = None) -> Image.Image:
        """Generate high-quality cover art using Stable Diffusion"""
        
        # Build prompt based on inputs
        if custom_prompt:
            prompt = custom_prompt
        else:
            emotion_styles = {
                'positive': 'vibrant, energetic, glowing, neon, holographic',
                'neutral': 'balanced, modern, tech-forward, sleek',
                'negative': 'intense, dramatic, cyberpunk, moody',
            }
            
            emotion_desc = emotion_styles.get(primary_emotion, emotion_styles['positive'])
            
            prompt = f"""
            Hyper-energetic Spotify Wrapped-style neon cyberpunk cover art for {title}.
            {emotion_desc} aesthetic. Holographic gradients, vaporwave ribbons, glowing geometric shapes.
            Dynamic motion-graphics, futuristic chat bubbles, luminous 3D UI shards.
            Cinematic framing, deep blacks with neon pink-blue-green streaks.
            AI theme, tech elements, professional design, ultra high quality.
            """
        
        try:
            # Generate image
            image = self.pipe(
                prompt=prompt,
                height=1024,
                width=1024,
                num_inference_steps=50,
                guidance_scale=7.5,
                num_images_per_prompt=1,
            ).images[0]
            
            # Enhance with text overlay
            return self._add_text_overlay(image, title, subtitle)
        except Exception as e:
            print(f"Error generating image: {e}")
            return self._create_fallback_cover(title, subtitle)
    
    def generate_style_variations(self, 
                                 image: Image.Image,
                                 styles: List[str] = None) -> Dict[str, Image.Image]:
        """Generate multiple style variations of an image"""
        
        if styles is None:
            styles = ['neon', 'hologram', 'glitch', 'plasma', 'ethereal']
        
        variations = {}
        
        for style in styles:
            variations[style] = self._apply_style(image, style)
        
        return variations
    
    def _add_text_overlay(self, 
                         image: Image.Image,
                         title: str,
                         subtitle: str) -> Image.Image:
        """Add glowing text overlay to image"""
        
        draw = ImageDraw.Draw(image, 'RGBA')
        
        # Add shadow/glow effect
        for offset in range(5, 0, -1):
            alpha = int(50 * (1 - offset / 5))
            draw.text(
                (512 - len(title) * 3 + offset, 200 + offset),
                title,
                fill=(255, 0, 110, alpha),
                font=None
            )
        
        # Main text
        draw.text(
            (512 - len(title) * 3, 200),
            title,
            fill=(255, 255, 255, 255),
            font=None
        )
        
        # Subtitle
        draw.text(
            (512 - len(subtitle) * 2, 900),
            subtitle,
            fill=(0, 217, 255, 200),
            font=None
        )
        
        return image
    
    def _apply_style(self, image: Image.Image, style: str) -> Image.Image:
        """Apply artistic style to image"""
        
        # Convert to array
        img_array = np.array(image, dtype=np.float32) / 255.0
        
        if style == 'neon':
            # Boost contrast and saturation
            img_array = np.power(img_array, 0.7) * 1.3
        elif style == 'hologram':
            # Add color shifting effect
            img_array[:, :, 0] *= 1.2
            img_array[:, :, 2] *= 0.8
        elif style == 'glitch':
            # Add glitch effect
            h, w = img_array.shape[:2]
            shift = np.random.randint(1, w // 10)
            img_array[:, shift:] = img_array[:, :-shift]
        elif style == 'plasma':
            # Add plasma effect
            img_array = np.clip(img_array * 1.5, 0, 1)
        elif style == 'ethereal':
            # Soft glow effect
            from scipy.ndimage import gaussian_filter
            img_array = gaussian_filter(img_array, sigma=5)
        
        # Convert back to image
        img_array = np.clip(img_array * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(img_array)
    
    def _create_fallback_cover(self, title: str, subtitle: str) -> Image.Image:
        """Create fallback cover with pure gradient and text"""
        
        image = Image.new('RGB', (1024, 1024), color=(5, 6, 18))
        draw = ImageDraw.Draw(image)
        
        # Draw gradient-like background
        for y in range(1024):
            ratio = y / 1024
            r = int(5 + (255 - 5) * ratio)
            g = int(6 + (200 - 6) * ratio)
            b = int(18 + (255 - 18) * ratio)
            draw.line([(0, y), (1024, y)], fill=(r, g, b))
        
        # Add neon text
        draw.text((100, 400), title, fill=(255, 0, 110))
        draw.text((100, 900), subtitle, fill=(0, 217, 255))
        
        return image
    
    def generate_animated_sequence(self, 
                                  base_prompt: str,
                                  frames: int = 12) -> List[Image.Image]:
        """Generate animated sequence of images"""
        
        sequence = []
        
        for i in range(frames):
            # Vary prompt slightly for each frame
            modified_prompt = f"{base_prompt} frame {i+1} of {frames}"
            
            image = self.pipe(
                prompt=modified_prompt,
                height=512,
                width=512,
                num_inference_steps=30,
                guidance_scale=7.5,
            ).images[0]
            
            sequence.append(image)
        
        return sequence
