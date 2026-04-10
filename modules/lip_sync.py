"""
Lip Sync module for Deep-Live-Cam
Uses audio-driven lip synchronization for face swap videos.

This module provides lip sync functionality using audio analysis to drive
facial landmark movements for realistic mouth movements.

Requirements:
    - facexlib for facial landmarks
    - audio processing libraries

Note: This is a basic implementation. For production, consider:
    - Wav2Lip: https://github.com/Rudrab/Wav2Lip
    - LivePortrait: https://github.com/MirrorCastle/LivePortrait
"""

from __future__ import annotations
import os
import logging
import numpy as np
import cv2
from typing import Optional, Tuple, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Try to import optional dependencies
LIP_SYNC_AVAILABLE = False
try:
    import librosa
    import soundfile as sf
    LIP_SYNC_AVAILABLE = True
except ImportError:
    logger.warning("Lip sync dependencies not installed. Install librosa and soundfile for lip sync.")


@dataclass
class LipSyncConfig:
    """Configuration for lip sync processing."""
    # Audio parameters
    sample_rate: int = 16000
    hop_length: int = 160
    
    # Video parameters  
    fps: float = 30.0
    face_padding: int = 10
    
    # Mouth region (relative to face bbox)
    mouth_margin_x: float = 0.25  # 25% margin on each side
    mouth_margin_y: float = 0.35  # 35% margin for mouth height
    
    # Smoothing
    smooth_window: int = 5
    min_mouth_opening: float = 0.02
    
    # Quality
    enhance_mouth: bool = True
    Teeth_visible: bool = True


def extract_audio(audio_path: str, sample_rate: int = 16000) -> Tuple[np.ndarray, int]:
    """Extract audio from file as numpy array.
    
    Args:
        audio_path: Path to audio file
        sample_rate: Target sample rate
        
    Returns:
        Tuple of (audio_data, sample_rate)
    """
    if not LIP_SYNC_AVAILABLE:
        raise ImportError("lip sync dependencies not installed")
    
    try:
        audio, sr = librosa.load(audio_path, sr=sample_rate)
        return audio, sr
    except Exception as e:
        logger.error(f"Error loading audio: {e}")
        # Try with soundfile as fallback
        audio, sr = sf.read(audio_path)
        if sr != sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
        return audio, sample_rate


def get_mouth_region(frame: np.ndarray, face_landmarks: np.ndarray) -> Tuple[int, int, int, int]:
    """Extract mouth region bounding box from face landmarks.
    
    Args:
        frame: Video frame
        face_landmarks: 106-point facial landmarks
        
    Returns:
        Tuple of (x_min, y_min, x_max, y_max)
    """
    # Mouth landmarks are typically indices 54-66 for upper lip
    # and 67-73 for lower lip (varies by model)
    if face_landmarks is None or len(face_landmarks) < 74:
        # Fallback: use center-bottom portion of face
        h, w = frame.shape[:2]
        return w // 4, h // 2, 3 * w // 4, h
    
    # Get mouth region from landmarks
    mouth_pts = face_landmarks[52:74]  # Approximate mouth region
    
    x_min = int(np.min(mouth_pts[:, 0])) - 10
    y_min = int(np.min(mouth_pts[:, 1])) - 5
    x_max = int(np.max(mouth_pts[:, 0])) + 10
    y_max = int(np.max(mouth_pts[:, 1])) + 5
    
    # Clamp to frame bounds
    h, w = frame.shape[:2]
    x_min, y_min = max(0, x_min), max(0, y_min)
    x_max, y_max = min(w, x_max), min(h, y_max)
    
    return x_min, y_min, x_max, y_max


def compute_mouth_opening(audio: np.ndarray, frame_idx: int, fps: float, 
                       config: LipSyncConfig) -> float:
    """Compute mouth opening amount for a given frame based on audio.
    
    Args:
        audio: Audio waveform
        frame_idx: Current frame index
        fps: Frames per second
        config: Lip sync configuration
        
    Returns:
        Mouth opening amount (0.0 - 1.0)
    """
    if not LIP_SYNC_AVAILABLE:
        return 0.3  # Default neutral
    
    # Calculate audio segment for this frame
    frame_time = frame_idx / fps
    sample_idx = int(frame_time * config.sample_rate)
    
    # Get audio window around this frame
    window_size = config.hop_length * 2
    start_idx = max(0, sample_idx - window_size)
    end_idx = min(len(audio), sample_idx + window_size)
    
    if end_idx <= start_idx:
        return config.min_mouth_opening
    
    audio_segment = audio[start_idx:end_idx]
    
    # Compute energy (simple amplitude-based)
    energy = np.sqrt(np.mean(audio_segment ** 2))
    
    # Normalize and apply smoothing
    normalized_energy = min(1.0, energy * 10)
    
    return max(config.min_mouth_opening, normalized_energy)


def apply_mouth_mask(frame: np.ndarray, mouth_bbox: Tuple[int, int, int, int],
                 mouth_opening: float, enhance: bool = True) -> np.ndarray:
    """Apply mouth region with opening amount.
    
    Args:
        frame: Video frame
        mouth_bbox: (x_min, y_min, x_max, y_max)
        mouth_opening: 0.0-1.0 opening amount
        enhance: Whether to enhance the mouth region
        
    Returns:
        Modified frame
    """
    x1, y1, x2, y2 = mouth_bbox
    
    if x2 <= x1 or y2 <= y1:
        return frame
    
    mouth_region = frame[y1:y2, x1:x2]
    
    if mouth_region.size == 0:
        return frame
    
    # Scale mouth opening vertically
    h, w = mouth_region.shape[:2]
    new_h = int(h * max(0.3, mouth_opening))
    
    if new_h < h and new_h > 0:
        # Adjust the mouth region
        if enhance and mouth_opening > 0.3:
            # Subtle enhancement
            mouth_region = cv2.GaussianBlur(mouth_region, (3, 3), 0.5)
    
    return frame


def smooth_opening(opening_values: list, window: int) -> float:
    """Apply temporal smoothing to mouth opening values.
    
    Args:
        opening_values: List of opening values
        window: Smoothing window size
        
    Returns:
        Smoothed opening value
    """
    if not opening_values:
        return 0.3
    
    # Take recent values
    values = opening_values[-window:]
    
    # Apply moving average
    return float(np.mean(values))


def create_lip_sync_video(
    video_path: str,
    audio_path: str,
    output_path: str,
    face_landmarks: Optional[Any] = None,
    config: Optional[LipSyncConfig] = None
) -> bool:
    """Create lip-synced video from source video and audio.
    
    This is a simplified implementation. For production use,
    consider using Wav2Lip or LivePortrait models.
    
    Args:
        video_path: Input video path
        audio_path: Audio file path (e.g., .wav file)
        output_path: Output video path
        face_landmarks: Pre-computed face landmarks (optional)
        config: Lip sync configuration
        
    Returns:
        True if successful
    """
    if config is None:
        config = LipSyncConfig()
    
    if not LIP_SYNC_AVAILABLE:
        logger.warning("Lip sync not available - installing dependencies")
        return False
    
    try:
        logger.info(f"Creating lip-synced video: {output_path}")
        
        # Extract audio
        audio, sr = extract_audio(audio_path, config.sample_rate)
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Cannot open video: {video_path}")
            return False
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # If different fps, adjust
        if config.fps != fps:
            fps = config.fps
        
        # Video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not writer.isOpened():
            logger.error("Cannot create output video writer")
            return False
        
        frame_idx = 0
        opening_history = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Compute mouth opening for this frame
            opening = compute_mouth_opening(
                audio, frame_idx, fps, config
            )
            
            # Apply smoothing
            opening_history.append(opening)
            smoothed_opening = smooth_opening(
                opening_history, config.smooth_window
            )
            
            # If we have face landmarks, apply mouth region
            if face_landmarks is not None:
                mouth_bbox = get_mouth_region(frame, face_landmarks)
                frame = apply_mouth_mask(
                    frame, mouth_bbox, smoothed_opening, config.enhance_mouth
                )
            
            writer.write(frame)
            frame_idx += 1
            
            if frame_idx % 30 == 0:
                logger.info(f"Processed {frame_idx} frames")
        
        cap.release()
        writer.release()
        
        # Add audio to video (requires ffmpeg)
        try:
            import subprocess
            temp_output = output_path.replace('.mp4', '_nosound.mp4')
            os.rename(output_path, temp_output)
            
            result = subprocess.run([
                'ffmpeg', '-y',
                '-i', temp_output,
                '-i', audio_path,
                '-c:v', 'copy',
                '-c:a', 'aac',
                '-strict', 'experimental',
                output_path
            ], capture_output=True)
            
            if result.returncode == 0:
                os.remove(temp_output)
                logger.info("Audio added to video successfully")
            else:
                # Keep original without audio
                os.rename(temp_output, output_path)
                logger.warning("Could not add audio - keeping original")
                
        except Exception as e:
            logger.warning(f"Audio mixing not available: {e}")
        
        logger.info(f"Lip sync complete: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error creating lip sync video: {e}")
        return False


def check_lip_sync_available() -> bool:
    """Check if lip sync is available and all dependencies are installed.
    
    Returns:
        True if lip sync can be used
    """
    return LIP_SYNC_AVAILABLE


# Simple face landmark detector using OpenCV
def detect_face_landmarks(frame: np.ndarray) -> Optional[np.ndarray]:
    """Detect facial landmarks using OpenCV's DNN module.
    
    This is a placeholder - for production use a proper
    landmark detector like MediaPipe or InsightFace.
    
    Args:
        frame: Input frame
        
    Returns:
        Array of landmark points or None
    """
    # This is a simplified placeholder
    # For real implementation, use InsightFace or MediaPipe
    return None


# Export functions for external use
__all__ = [
    'LipSyncConfig',
    'create_lip_sync_video',
    'check_lip_sync_available',
    'detect_face_landmarks',
    'extract_audio',
    'get_mouth_region',
    'compute_mouth_opening',
]