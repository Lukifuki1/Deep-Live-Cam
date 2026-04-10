"""
Batch Processing module for Deep-Live-Cam

This module provides batch processing capabilities for processing
multiple images or videos at once.

Usage:
    from modules.batch import batch_process_files
    
    results = batch_process_files(
        source_image = "source.jpg",
        targets = ["video1.mp4", "video2.mp4", "image.jpg"],
        output_dir = "output/",
        keep_fps = True,
        keep_audio = True
    )
"""

from __future__ import annotations
import os
import logging
import shutil
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

import cv2
import modules.globals
from modules.utilities import is_image, is_video, has_image_extension

logger = logging.getLogger(__name__)

# Batch processing configuration
MAX_BATCH_WORKERS = 4  # Number of parallel workers
MAX_BATCH_SIZE = 100  # Maximum files to process at once


class BatchProcessConfig:
    """Configuration for batch processing."""
    
    def __init__(
        self,
        num_workers: int = MAX_BATCH_WORKERS,
        skip_existing: bool = True,
        recursive: bool = False,
        output_format: str = "same",  # "same", "mp4", "jpg", "png"
        quality: int = 23,  # CRF quality (lower = better)
        keep_fps: bool = True,
        keep_audio: bool = True,
    ):
        self.num_workers = num_workers
        self.skip_existing = skip_existing
        self.recursive = recursive
        self.output_format = output_format
        self.quality = quality
        self.keep_fps = keep_fps
        self.keep_audio = keep_audio


class BatchResult:
    """Result of batch processing."""
    
    def __init__(
        self,
        input_path: str,
        output_path: Optional[str],
        success: bool,
        error: Optional[str] = None,
        processing_time: float = 0.0
    ):
        self.input_path = input_path
        self.output_path = output_path
        self.success = success
        self.error = error
        self.processing_time = processing_time
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "input": self.input_path,
            "output": self.output_path,
            "success": self.success,
            "error": self.error,
            "time": self.processing_time
        }


def get_files_from_directory(
    directory: str,
    extensions: Optional[List[str]] = None,
    recursive: bool = False
) -> List[str]:
    """Get all image/video files from a directory.
    
    Args:
        directory: Input directory
        extensions: List of extensions to include (None = all)
        recursive: Whether to search recursively
        
    Returns:
        List of file paths
    """
    if extensions is None:
        extensions = [
            '*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp',
            '*.mp4', '*.mkv', '*.avi', '*.mov'
        ]
    
    files = []
    directory_path = Path(directory)
    
    if recursive:
        for ext in extensions:
            files.extend(directory_path.rglob(ext))
    else:
        for ext in extensions:
            files.extend(directory_path.glob(ext))
    
    return [str(f) for f in files]


def resolve_output_path(
    input_path: str,
    output_dir: str,
    output_format: str = "same"
) -> str:
    """Resolve output path for a file.
    
    Args:
        input_path: Input file path
        output_dir: Output directory
        output_format: "same", "mp4", "jpg", or "png"
        
    Returns:
        Output file path
    """
    input_file = Path(input_path)
    stem = input_file.stem
    ext = input_file.suffix.lower()
    
    if output_format == "same":
        new_ext = ext
    elif output_format == "mp4":
        new_ext = ".mp4"
    elif output_format == "jpg":
        new_ext = ".jpg"
    elif output_format == "png":
        new_ext = ".png"
    else:
        new_ext = ext
    
    return str(Path(output_dir) / f"{stem}_output{new_ext}")


def process_single_file(
    source_path: str,
    target_path: str,
    output_path: str,
    frame_processor: str = "face_swapper",
    keep_fps: bool = True,
    keep_audio: bool = True,
    quality: int = 23
) -> BatchResult:
    """Process a single source-target pair.
    
    Note: This is a placeholder for the actual processing logic.
    The real implementation would call the core processing functions.
    
    Args:
        source_path: Source face image
        target_path: Target image/video
        output_path: Output file path
        frame_processor: Frame processor to use
        keep_fps: Whether to keep original FPS
        keep_audio: Whether to keep audio
        quality: Output quality (CRF)
        
    Returns:
        BatchResult
    """
    import time
    import modules.globals as globals
    
    start_time = time.time()
    
    try:
        # Set up processing
        modules.globals.source_path = source_path
        modules.globals.target_path = target_path
        modules.globals.output_path = output_path
        modules.globals.frame_processors = [frame_processor]
        modules.globals.keep_fps = keep_fps
        modules.globals.keep_audio = keep_audio
        modules.globals.video_quality = quality
        modules.globals.headless = True
        
        # In a real implementation, this would call the core processing
        # For now, we just copy the file as a placeholder
        if is_image(target_path):
            shutil.copy2(target_path, output_path)
        else:
            # For video, we'd need actual processing
            # Just copy as placeholder
            with open(target_path, 'rb') as src:
                with open(output_path, 'wb') as dst:
                    dst.write(src.read())
        
        processing_time = time.time() - start_time
        
        return BatchResult(
            input_path=target_path,
            output_path=output_path,
            success=True,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Error processing {target_path}: {e}")
        return BatchResult(
            input_path=target_path,
            output_path=output_path,
            success=False,
            error=str(e),
            processing_time=time.time() - start_time
        )


def batch_process_files(
    source_path: str,
    targets: List[str],
    output_dir: str,
    output_format: str = "same",
    config: Optional[BatchProcessConfig] = None
) -> List[BatchResult]:
    """Process multiple files with the same source face.
    
    Args:
        source_path: Source face image path
        targets: List of target file paths
        output_dir: Output directory
        output_format: Output format ("same", "mp4", "jpg", "png")
        config: Batch processing configuration
        
    Returns:
        List of BatchResults
    """
    if config is None:
        config = BatchProcessConfig()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    
    logger.info(f"Processing {len(targets)} files with {source_path}")
    
    # Process files
    with ThreadPoolExecutor(max_workers=config.num_workers) as executor:
        # Submit all jobs
        futures = {}
        for target_path in targets:
            output_path = resolve_output_path(
                target_path, output_dir, output_format
            )
            
            # Skip if exists
            if config.skip_existing and os.path.exists(output_path):
                results.append(BatchResult(
                    input_path=target_path,
                    output_path=output_path,
                    success=True,
                    error="skipped_existing"
                ))
                continue
            
            future = executor.submit(
                process_single_file,
                source_path,
                target_path,
                output_path,
                keep_fps=config.keep_fps,
                keep_audio=config.keep_audio,
                quality=config.quality
            )
            futures[future] = target_path
        
        # Collect results
        for future in as_completed(futures):
            target_path = futures[future]
            try:
                result = future.result()
                results.append(result)
                
                if result.success:
                    logger.info(f"✓ Processed: {target_path}")
                else:
                    logger.error(f"✗ Failed: {target_path} - {result.error}")
                    
            except Exception as e:
                logger.error(f"✗ Error: {target_path} - {e}")
                results.append(BatchResult(
                    input_path=target_path,
                    output_path=None,
                    success=False,
                    error=str(e)
                ))
    
    # Summary
    success_count = sum(1 for r in results if r.success)
    logger.info(f"Batch complete: {success_count}/{len(results)} successful")
    
    return results


def batch_process_directory(
    source_path: str,
    input_dir: str,
    output_dir: str,
    config: Optional[BatchProcessConfig] = None
) -> List[BatchResult]:
    """Process all files in a directory with a single source.
    
    Args:
        source_path: Source face image
        input_dir: Input directory with target files
        output_dir: Output directory
        config: Batch processing config
        
    Returns:
        List of BatchResults
    """
    if config is None:
        config = BatchProcessConfig()
    
    # Get all files from input directory
    targets = get_files_from_directory(
        input_dir,
        recursive=config.recursive
    )
    
    if not targets:
        logger.warning(f"No files found in {input_dir}")
        return []
    
    logger.info(f"Found {len(targets)} files in {input_dir}")
    
    return batch_process_files(
        source_path=source_path,
        targets=targets,
        output_dir=output_dir,
        output_format=config.output_format,
        config=config
    )


def find_duplicates(directory: str) -> Dict[str, List[str]]:
    """Find duplicate images by content hash.
    
    Args:
        directory: Directory to scan
        
    Returns:
        Dictionary mapping hash to file paths
    """
    import hashlib
    
    hashes = {}
    
    files = get_files_from_directory(directory)
    
    for file_path in files:
        try:
            # Compute hash
            with open(file_path, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
            
            if file_hash in hashes:
                hashes[file_hash].append(file_path)
            else:
                hashes[file_hash] = [file_path]
                
        except Exception as e:
            logger.warning(f"Error hashing {file_path}: {e}")
    
    # Filter to only duplicates
    duplicates = {k: v for k, v in hashes.items() if len(v) > 1}
    
    return duplicates


def batch_rename(
    directory: str,
    pattern: str = "{index:03d}_{name}",
    start_index: int = 0
) -> int:
    """Batch rename files in a directory.
    
    Args:
        directory: Directory with files
        pattern: Rename pattern (supports {index}, {name}, {ext})
        start_index: Starting index
        
    Returns:
        Number of files renamed
    """
    files = get_files_from_directory(directory)
    
    index = start_index
    for file_path in files:
        file = Path(file_path)
        new_name = pattern.format(
            index=index,
            name=file.stem,
            ext=file.suffix
        )
        
        new_path = os.path.join(file.parent, new_name)
        
        try:
            os.rename(file_path, new_path)
            index += 1
        except Exception as e:
            logger.error(f"Error renaming {file_path}: {e}")
    
    logger.info(f"Renamed {index - start_index} files")
    return index - start_index


# Report generation
def generate_batch_report(
    results: List[BatchResult],
    output_path: Optional[str] = None
) -> str:
    """Generate a report from batch results.
    
    Args:
        results: List of batch results
        output_path: Optional output file path
        
    Returns:
        Report text
    """
    import json
    from datetime import datetime
    
    total = len(results)
    successful = sum(1 for r in results if r.success)
    failed = total - successful
    total_time = sum(r.processing_time for r in results)
    avg_time = total_time / total if total > 0 else 0
    
    report = f"""# Batch Processing Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary
- Total files: {total}
- Successful: {successful}
- Failed: {failed}
- Success rate: {successful/total*100:.1f}%
- Total time: {total_time:.2f}s
- Average time: {avg_time:.2f}s per file

## Files
"""
    
    # Add each result
    for result in results:
        status = "✓" if result.success else "✗"
        report += f"- {status} {Path(result.input_path).name}"
        if result.error:
            report += f" ({result.error})"
        report += f" - {result.processing_time:.2f}s\n"
    
    # Save to file
    if output_path:
        with open(output_path, 'w') as f:
            f.write(report)
    
    return report


# Export
__all__ = [
    'BatchProcessConfig',
    'BatchResult',
    'batch_process_files',
    'batch_process_directory',
    'find_duplicates',
    'batch_rename',
    'generate_batch_report',
    'get_files_from_directory',
]