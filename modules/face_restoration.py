"""
Face Restoration module for Deep-Live-Cam
Provides face quality enhancement using GFPGAN and CodeFormer models.

This module enhances face quality after face swap for more realistic results.
Models are automatically downloaded on first use.

Models:
- GFPGAN: https://github.com/TencentARC/GFPGAN
- CodeFormer: https://github.com/sczhou/CodeFormer
"""

from __future__ import annotations
import os
import logging
from typing import Optional, Tuple
from pathlib import Path

import cv2
import numpy as np
import modules.globals
from modules.gpu_processing import gpu_resize, gpu_cvt_color

logger = logging.getLogger(__name__)

# Module paths
RESTORATION_MODELS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models", "restoration"
)

# GFPGAN model URLs (from HuggingFace)
GFPGAN_MODELS = {
    "gfpgan": {
        "url": "https://huggingface.co/hacksider/deep-live-cam/resolve/main/GFPGAN.pth",
        "name": "GFPGAN.pth"
    }
}

# CodeFormer model URLs
CODEFORMER_MODELS = {
    "codeformer": {
        "url": "https://huggingface.co/hacksider/deep-live-cam/resolve/main/CodeFormer.pth", 
        "name": "CodeFormer.pth"
    }
}


class FaceRestorer:
    """Face restoration processor using GFPGAN/CodeFormer."""
    
    def __init__(self, model_type: str = "gfpgan"):
        """Initialize face restorer.
        
        Args:
            model_type: "gfpgan" or "codeformer"
        """
        self.model_type = model_type
        self.model = None
        self.device = None
        self._model_loaded = False
    
    def _download_model(self, model_path: str) -> bool:
        """Download model if not exists."""
        if os.path.exists(model_path):
            return True
        
        # Try to download
        try:
            from modules.utilities import conditional_download
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            if self.model_type == "gfpgan":
                urls = [GFPGAN_MODELS["gfpgan"]["url"]]
            else:
                urls = [CODEFORMER_MODELS["codeformer"]["url"]]
            
            conditional_download(os.path.dirname(model_path), urls)
            return os.path.exists(model_path)
        except Exception as e:
            logger.error(f"Failed to download model: {e}")
            return False
    
    def load_model(self) -> bool:
        """Load the restoration model."""
        if self._model_loaded:
            return True
        
        try:
            # Try to load GFPGAN
            if self.model_type == "gfpgan":
                try:
                    from gfpgan import GFPGANer
                    
                    model_path = os.path.join(
                        RESTORATION_MODELS_DIR, 
                        GFPGAN_MODELS["gfpgan"]["name"]
                    )
                    
                    # Download if not exists
                    self._download_model(model_path)
                    
                    # Determine device
                    if 'CUDAExecutionProvider' in modules.globals.execution_providers:
                        self.device = 'cuda'
                    else:
                        self.device = 'cpu'
                    
                    # Check if model exists, if not skip
                    if not os.path.exists(model_path):
                        logger.warning(f"GFPGAN model not found at {model_path}")
                        return False
                    
                    self.model = GFPGANer(
                        model_path=model_path,
                        upscale=1,
                        device=self.device
                    )
                    self._model_loaded = True
                    logger.info("GFPGAN model loaded successfully")
                    return True
                    
                except ImportError:
                    logger.warning("GFPGAN not installed - face restoration unavailable")
                    return False
                    
            elif self.model_type == "codeformer":
                try:
                    from codeformer import CodeFormerer
                    
                    model_path = os.path.join(
                        RESTORATION_MODELS_DIR,
                        CODEFORMER_MODELS["codeformer"]["name"]
                    )
                    
                    self._download_model(model_path)
                    
                    if not os.path.exists(model_path):
                        logger.warning("CodeFormer model not found")
                        return False
                    
                    if 'CUDAExecutionProvider' in modules.globals.execution_providers:
                        self.device = 'cuda'
                    else:
                        self.device = 'cpu'
                    
                    self.model = CodeFormerer(
                        model_path=model_path,
                        device=self.device
                    )
                    self._model_loaded = True
                    logger.info("CodeFormer model loaded successfully")
                    return True
                    
                except ImportError:
                    logger.warning("CodeFormer not installed")
                    return False
                    
        except Exception as e:
            logger.error(f"Failed to load face restorer: {e}")
            return False
        
        return False
    
    def enhance(self, image: np.ndarray) -> np.ndarray:
        """Enhance face quality.
        
        Args:
            image: Input face image (BGR format)
            
        Returns:
            Enhanced face image
        """
        if not self._model_loaded:
            if not self.load_model():
                return image
        
        if self.model is None:
            return image
        
        try:
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Apply restoration
            if self.model_type == "gfpgan":
                _, output, _ = self.model.process(
                    image_rgb, 
                    has_aligned=False, 
                    only_center_face=False,
                    paste_back=True
                )
            else:
                output = self.model.process(
                    image_rgb,
                    fidelity_weight=0.5
                )
            
            if output is None:
                return image
            
            # Convert back to BGR
            output_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
            return output_bgr
            
        except Exception as e:
            logger.error(f"Error enhancing face: {e}")
            return image
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Process face image (convenience method)."""
        return self.enhance(image)


# Global instances
_gfpgan_restorer: Optional[FaceRestorer] = None
_codeformer_restorer: Optional[FaceRestorer] = None


def get_gfpgan_restorer() -> Optional[FaceRestorer]:
    """Get GFPGAN restorer instance."""
    global _gfpgan_restorer
    if _gfpgan_restorer is None:
        _gfpgan_restorer = FaceRestorer("gfpgan")
        _gfpgan_restorer.load_model()
    return _gfpgan_restorer


def get_codeformer_restorer() -> Optional[FaceRestorer]:
    """Get CodeFormer restorer instance."""
    global _codeformer_restorer
    if _codeformer_restorer is None:
        _codeformer_restorer = FaceRestorer("codeformer")
        _codeformer_restorer.load_model()
    return _codeformer_restorer


def enhance_face(
    face_image: np.ndarray, 
    model_type: str = "gfpgan"
) -> np.ndarray:
    """Enhance a face image.
    
    Args:
        face_image: Input face (BGR)
        model_type: "gfpgan" or "codeformer"
        
    Returns:
        Enhanced face image
    """
    if model_type == "gfpgan":
        restorer = get_gfpgan_restorer()
    else:
        restorer = get_codeformer_restorer()
    
    if restorer is None:
        return face_image
    
    return restorer.enhance(face_image)


def _simple_enhance(image: np.ndarray) -> np.ndarray:
    """Simple face enhancement using OpenCV (fallback if models not available).
    
    Args:
        image: Input face image
        
    Returns:
        Enhanced face
    """
    # Apply bilateral filter for edge-preserving smoothing
    enhanced = cv2.bilateralFilter(image, 9, 75, 75)
    
    # Apply unsharp masking for sharpness
    blurred = cv2.GaussianBlur(enhanced, (0, 0), 3.0)
    sharpened = cv2.addWeighted(enhanced, 1.5, blurred, -0.5, 0)
    
    # Slight contrast enhancement
    lab = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.equalizeHist(l)
    enhanced_lab = cv2.merge([l, a, b])
    result = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    
    return result


def enhance_frame(
    frame: np.ndarray,
    face_bbox: Optional[Tuple[int, int, int, int]] = None,
    model_type: str = "gfpgan"
) -> np.ndarray:
    """Enhance faces in a frame.
    
    Args:
        frame: Input video frame
        face_bbox: Optional face bounding box (x1, y1, x2, y2)
        model_type: "gfpgan", "codeformer", or "simple"
        
    Returns:
        Enhanced frame
    """
    if face_bbox is None:
        # Use simple enhancement on full frame
        if model_type == "simple":
            return _simple_enhance(frame)
        return enhance_face(frame, model_type)
    
    x1, y1, x2, y2 = face_bbox
    
    # Extract face region
    face = frame[y1:y2, x1:x2]
    if face.size == 0:
        return frame
    
    # Enhance face
    if model_type == "simple":
        enhanced_face = _simple_enhance(face)
    else:
        enhanced_face = enhance_face(face, model_type)
    
    # Put back
    frame[y1:y2, x1:x2] = enhanced_face
    
    return frame


# Check if restoration is available
def check_restoration_available(model_type: str = "gfpgan") -> bool:
    """Check if face restoration is available.
    
    Args:
        model_type: "gfpgan" or "codeformer"
        
    Returns:
        True if model can be loaded
    """
    try:
        if model_type == "gfpgan":
            from gfpgan import GFPGANer
        else:
            from codeformer import CodeFormerer
        return True
    except ImportError:
        return False


# Resolution helpers
def get_resolution_size(resolution: str) -> Tuple[int, int]:
    """Get target resolution size.
    
    Args:
        resolution: "original", "1080p", "2k", "4k", "8k"
        
    Returns:
        (width, height)
    """
    sizes = {
        "original": (0, 0),
        "1080p": (1920, 1080),
        "2k": (2560, 1440),
        "4k": (3840, 2160),
        "8k": (7680, 4320)
    }
    return sizes.get(resolution, (0, 0))


def resize_to_resolution(
    frame: np.ndarray, 
    target_resolution: str
) -> np.ndarray:
    """Resize frame to target resolution.
    
    Args:
        frame: Input frame
        target_resolution: "original", "1080p", "2k", "4k", "8k"
        
    Returns:
        Resized frame
    """
    if target_resolution == "original":
        return frame
    
    target_size = get_resolution_size(target_resolution)
    if target_size == (0, 0):
        return frame
    
    current_h, current_w = frame.shape[:2]
    target_w, target_h = target_size
    
    # Scale to fit within target while maintaining aspect ratio
    scale = min(target_w / current_w, target_h / current_h)
    if scale >= 1.0:
        return frame
    
    new_w = int(current_w * scale)
    new_h = int(current_h * scale)
    
    return gpu_resize(frame, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)


__all__ = [
    'FaceRestorer',
    'get_gfpgan_restorer',
    'get_codeformer_restorer', 
    'enhance_face',
    'enhance_frame',
    'check_restoration_available',
    'resize_to_resolution',
    'get_resolution_size',
]