"""
Web API module for Deep-Live-Cam
Provides REST API for remote access and processing.

This module allows controlling Deep-Live-Cam from other applications
or for building web interfaces.

Dependencies:
    - flask or fastapi

Usage:
    # Start API server:
    from modules.web_api import start_api_server
    start_api_server(host="0.0.0.0", port=5000)
    
    # Or run from command line:
    # python run.py --api --api-host 0.0.0.0 --api-port 5000
"""

from __future__ import annotations
import os
import sys
import logging
import json
import base64
from typing import Optional, Dict, Any, List
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check for available web frameworks
WEB_FRAMEWORK = None
try:
    from flask import Flask, request, jsonify, send_file
    WEB_FRAMEWORK = "flask"
except ImportError:
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import FileResponse, JSONResponse
        WEB_FRAMEWORK = "fastapi"
    except ImportError:
        logger.warning("No web framework available. Install flask or fastapi.")


class DeepLiveCamAPI:
    """REST API for Deep-Live-Cam."""
    
    def __init__(self, host: str = "0.0.0.0", port: int = 5000):
        """Initialize API server.
        
        Args:
            host: Host to bind to
            port: Port to listen on
        """
        self.host = host
        self.port = port
        self.app = None
        self.processing = False
        
        # Create app
        if WEB_FRAMEWORK == "flask":
            self._create_flask_app()
        elif WEB_FRAMEWORK == "fastapi":
            self._create_fastapi_app()
        else:
            raise RuntimeError("No web framework available")
    
    def _create_flask_app(self):
        """Create Flask application."""
        from flask import Flask, request, jsonify, send_file
        
        self.app = Flask(__name__)
        
        # Health check
        @self.app.route('/health')
        def health():
            return jsonify({"status": "healthy", "framework": "flask"})
        
        @self.app.route('/api/status')
        def status():
            return jsonify({
                "status": "ready" if not self.processing else "processing",
                "framework": "flask"
            })
        
        # Process endpoint
        @self.app.route('/api/process', methods=['POST'])
        def process():
            return self._handle_process()
        
        # Batch process
        @self.app.route('/api/batch', methods=['POST'])
        def batch():
            return self._handle_batch()
        
        # Get models
        @self.app.route('/api/models')
        def models():
            return jsonify(self._get_models())
        
        # Process video
        @self.app.route('/api/process_video', methods=['POST'])
        def process_video():
            return self._handle_process_video()
        
        # Cancel
        @self.app.route('/api/cancel', methods=['POST'])
        def cancel():
            self.processing = False
            return jsonify({"status": "cancelled"})
    
    def _create_fastapi_app(self):
        """Create FastAPI application."""
        from fastapi import FastAPI, HTTPException, UploadFile, File
        from fastapi.responses import JSONResponse, FileResponse
        from pydantic import BaseModel
        
        self.app = FastAPI(title="Deep-Live-Cam API")
        
        # Request models
        class ProcessRequest(BaseModel):
            source_image: str  # base64
            target_image: str   # base64
            many_faces: bool = False
        
        class VideoRequest(BaseModel):
            source_image: str
            target_video: str
            keep_fps: bool = True
        
        @self.app.get("/health")
        def health():
            return {"status": "healthy", "framework": "fastapi"}
        
        @self.app.get("/api/status")
        def status():
            return {"status": "ready" if not self.processing else "processing"}
        
        @self.app.post("/api/process")
        def process(req: ProcessRequest):
            return self._handle_process_fast(req)
        
        @self.app.post("/api/batch")
        def batch(req: List[ProcessRequest]):
            return self._handle_batch_fast(req)
        
        @self.app.get("/api/models")
        def models():
            return self._get_models()
    
    def _get_models(self) -> Dict[str, Any]:
        """Get available models."""
        return {
            "face_swapper": ["inswapper_128.onnx"],
            "face_enhancer": ["face_enhancer", "gpen256", "gpen512"],
            "execution_providers": ["CUDA", "DirectML", "CoreML", "CPU"]
        }
    
    def _handle_process(self) -> Dict[str, Any]:
        """Handle single image processing."""
        # This is a simplified version - full implementation would
        # handle the actual processing
        return {"status": "not_implemented", "message": "Use CLI for processing"}
    
    def _handle_process_video(self) -> Dict[str, Any]:
        """Handle video processing."""
        return {"status": "not_implemented", "message": "Use CLI for video processing"}
    
    def _handle_batch(self) -> Dict[str, Any]:
        """Handle batch processing."""
        return {"status": "not_implemented", "message": "Use batch module"}
    
    def _handle_process_fast(self, req) -> Dict[str, Any]:
        """Handle FastAPI process request."""
        return {"status": "not_implemented"}
    
    def _handle_batch_fast(self, req) -> Dict[str, Any]:
        """Handle FastAPI batch request."""
        return {"status": "not_implemented"}
    
    def run(self):
        """Start the API server."""
        if self.app is None:
            raise RuntimeError("No web app created")
        
        if WEB_FRAMEWORK == "flask":
            self.app.run(host=self.host, port=self.port, threaded=True)
        elif WEB_FRAMEWORK == "fastapi":
            import uvicorn
            uvicorn.run(self.app, host=self.host, port=self.port)


def start_api_server(host: str = "0.0.0.0", port: int = 5000) -> DeepLiveCamAPI:
    """Start the API server.
    
    Args:
        host: Host to bind to
        port: Port to listen on
        
    Returns:
        API server instance
    """
    api = DeepLiveCamAPI(host=host, port=port)
    logger.info(f"Starting API server on {host}:{port}")
    return api


# Simple HTTP server fallback (no external dependencies)
class SimpleAPI:
    """Simple HTTP API server without external dependencies."""
    
    def __init__(self, host: str = "0.0.0.0", port: int = 5000):
        self.host = host
        self.port = port
    
    def run(self):
        """Run simple HTTP server."""
        import http.server
        import socketserver
        
        class Handler(http.server.SimpleHTTPRequestHandler):
            def do_GET(self):
                if self.path == '/health':
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    self.wfile.write(b'{"status": "healthy"}')
                else:
                    super().do_GET()
            
            def do_POST(self):
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(b'{"status": "not_implemented"}')
        
        with socketserver.TCPServer((self.host, self.port), Handler) as httpd:
            logger.info(f"Serving on {self.host}:{self.port}")
            httpd.serve_forever()


def start_simple_api(host: str = "0.0.0.0", port: int = 5000):
    """Start simple API server without dependencies."""
    SimpleAPI(host=host, port=port).run()


# CLI argument handling
def parse_api_args():
    """Parse API-specific arguments."""
    import argparse
    
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--api', action='store_true', help='Start API server')
    parser.add_argument('--api-host', default='0.0.0.0', help='API host')
    parser.add_argument('--api-port', type=int, default=5000, help='API port')
    
    return parser.parse_args([])


# Export
__all__ = [
    'DeepLiveCamAPI',
    'start_api_server', 
    'start_simple_api',
    'parse_api_args',
]


# Entry point for running API
if __name__ == '__main__':
    args = parse_api_args()
    
    if args.api:
        try:
            start_api_server(host=args.api_host, port=args.api_port).run()
        except Exception as e:
            logger.error(f"Failed to start API: {e}")
            logger.info("Starting simple API fallback...")
            start_simple_api(host=args.api_host, port=args.api_port)
    else:
        logger.info("Use --api flag to start API server")