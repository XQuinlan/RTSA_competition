#!/usr/bin/env python3
"""
Raspberry Pi 5 Crowd Counting System
Real-time crowd counting with LWCC using Pi Camera
Optimized for edge deployment
"""

import cv2
import numpy as np
import torch
import time
import threading
import os
import psutil
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import logging
from typing import Tuple, Optional
import argparse

# Raspberry Pi Camera
try:
    from picamera2 import Picamera2
    PI_CAMERA_AVAILABLE = True
except ImportError:
    PI_CAMERA_AVAILABLE = False
    print("Warning: picamera2 not available. Using OpenCV fallback.")

# LWCC patching for local cache (from your original code)
import shutil
import os.path as osp

# Performance optimizations for Pi 5
os.environ['DNNL_DEFAULT_FPMATH_MODE'] = 'BF16'  # Enable BFloat16 if available
os.environ['LRU_CACHE_CAPACITY'] = '1024'
os.environ['THP_MEM_ALLOC_ENABLE'] = '1'  # Enable Transparent Huge Pages
os.environ['OMP_NUM_THREADS'] = '4'  # Optimize for Pi 5's CPU cores

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """Monitor system performance and inference times"""
    
    def __init__(self):
        self.inference_times = []
        self.memory_usage = []
        self.cpu_usage = []
    
    def record_inference_time(self, inference_time: float):
        self.inference_times.append(inference_time)
        if len(self.inference_times) > 100:  # Keep last 100 records
            self.inference_times.pop(0)
    
    def record_system_stats(self):
        memory = psutil.virtual_memory()
        cpu = psutil.cpu_percent()
        
        self.memory_usage.append(memory.percent)
        self.cpu_usage.append(cpu)
        
        if len(self.memory_usage) > 100:
            self.memory_usage.pop(0)
            self.cpu_usage.pop(0)
    
    def get_stats(self) -> dict:
        return {
            'avg_inference_time': np.mean(self.inference_times) if self.inference_times else 0,
            'avg_memory_usage': np.mean(self.memory_usage) if self.memory_usage else 0,
            'avg_cpu_usage': np.mean(self.cpu_usage) if self.cpu_usage else 0,
            'fps': 1.0 / np.mean(self.inference_times) if self.inference_times else 0
        }

class CrowdCounter:
    """Optimized crowd counting system for Raspberry Pi 5"""
    
    def __init__(self, model_name: str = "CSRNet", model_weights: str = "SHB", 
                 image_size: Tuple[int, int] = (224, 224), quantize: bool = True):
        
        self.model_name = model_name
        self.model_weights = model_weights
        self.image_size = image_size
        self.quantize = quantize
        
        # Performance monitoring
        self.perf_monitor = PerformanceMonitor()
        
        # Initialize camera
        self.camera = None
        self.cap = None
        
        # Setup local cache patching
        self._setup_lwcc_patches()
        
        # Load and optimize model
        self._load_model()
        
        logger.info(f"Crowd counter initialized with {model_name}-{model_weights}")
    
    def _setup_lwcc_patches(self):
        """Apply the same patches from your original code for local cache"""
        # Create local cache directory
        local_cache = os.path.expanduser("~/lwcc_cache")
        local_weights = os.path.join(local_cache, "weights")
        os.makedirs(local_weights, exist_ok=True)
        
        # Store original functions
        original_mkdir = Path.mkdir
        original_listdir = os.listdir
        original_move = shutil.move
        original_dirname = osp.dirname
        original_torch_load = torch.load
        
        def redirect_path(path):
            """Redirect /.lwcc paths to local cache"""
            path_str = str(path)
            if path_str.startswith("/.lwcc"):
                return path_str.replace("/.lwcc", local_cache)
            return path_str
        
        def patched_mkdir(self, mode=0o777, parents=False, exist_ok=False):
            redirected = redirect_path(self)
            if redirected != str(self):
                redirected_path = Path(redirected)
                return original_mkdir(redirected_path, mode, parents, exist_ok)
            return original_mkdir(self, mode, parents, exist_ok)
        
        def patched_listdir(path):
            redirected = redirect_path(path)
            return original_listdir(redirected)
        
        def patched_move(src, dst):
            redirected_dst = redirect_path(dst)
            return original_move(src, redirected_dst)
        
        def patched_dirname(path):
            result = original_dirname(path)
            return redirect_path(result)
        
        def patched_torch_load(f, map_location=None, pickle_module=None, **pickle_load_args):
            if isinstance(f, str):
                f = redirect_path(f)
            return original_torch_load(f, map_location, pickle_module, **pickle_load_args)
        
        # Apply patches
        Path.mkdir = patched_mkdir
        os.listdir = patched_listdir
        shutil.move = patched_move
        osp.dirname = patched_dirname
        torch.load = patched_torch_load
    
    def _load_model(self):
        """Load and optimize the LWCC model"""
        logger.info("Loading LWCC model...")
        
        # Set PyTorch optimization settings
        torch.backends.quantized.engine = 'qnnpack'  # ARM-optimized quantization
        torch.set_num_threads(4)  # Optimize for Pi 5's cores
        
        # Import LWCC after patching
        from lwcc import LWCC
        self.lwcc = LWCC
        
        # Pre-load model to avoid loading delays during inference
        try:
            # Load model once to cache it
            dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            _ = self.lwcc.get_count(dummy_image, model_name=self.model_name, 
                                   model_weights=self.model_weights, return_density=True)
            logger.info("Model loaded and cached successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def _init_camera(self):
        """Initialize camera (Pi Camera preferred, fallback to USB)"""
        if PI_CAMERA_AVAILABLE:
            try:
                self.camera = Picamera2()
                
                # Configure camera for our image size
                config = self.camera.create_preview_configuration(
                    main={"format": "RGB888", "size": self.image_size}
                )
                self.camera.configure(config)
                self.camera.start()
                
                logger.info("Pi Camera initialized successfully")
                return True
            except Exception as e:
                logger.warning(f"Pi Camera initialization failed: {e}")
        
        # Fallback to USB/OpenCV
        try:
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.image_size[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.image_size[1])
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            logger.info("OpenCV camera initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Camera initialization failed: {e}")
            return False
    
    def _capture_frame(self) -> Optional[np.ndarray]:
        """Capture frame from camera"""
        try:
            if self.camera is not None:
                # Pi Camera
                frame = self.camera.capture_array()
                return frame
            elif self.cap is not None:
                # OpenCV camera
                ret, frame = self.cap.read()
                if ret:
                    # Convert BGR to RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # Resize if needed
                    frame = cv2.resize(frame, self.image_size)
                    return frame
            return None
        except Exception as e:
            logger.error(f"Frame capture failed: {e}")
            return None
    
    def _count_people(self, image: np.ndarray) -> Tuple[float, np.ndarray]:
        """Perform crowd counting inference"""
        start_time = time.time()
        
        try:
            count, density = self.lwcc.get_count(
                image,
                model_name=self.model_name,
                model_weights=self.model_weights,
                return_density=True,
                resize_img=False  # We pre-resize images
            )
            
            inference_time = time.time() - start_time
            self.perf_monitor.record_inference_time(inference_time)
            
            return count, density
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            return 0.0, np.zeros((56, 56))  # Default density map size
    
    def _create_visualization(self, image: np.ndarray, count: float, 
                            density: np.ndarray, save_path: str = None) -> np.ndarray:
        """Create visualization with original image and density heatmap"""
        try:
            # Create density heatmap
            if density.max() > 0:
                normalized_density = (density / density.max() * 255).astype(np.uint8)
            else:
                normalized_density = np.zeros_like(density, dtype=np.uint8)
            
            # Apply colormap
            heat = cv2.applyColorMap(normalized_density, cv2.COLORMAP_JET)
            
            # Resize heatmap to match image size
            heat = cv2.resize(heat, self.image_size, interpolation=cv2.INTER_CUBIC)
            
            # Create side-by-side visualization
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Original image
            ax1.imshow(image)
            ax1.set_title(f"Original Image\nCount: {count:.1f}")
            ax1.axis('off')
            
            # Density heatmap
            ax2.imshow(heat)
            ax2.set_title("Density Heatmap")
            ax2.axis('off')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=100, bbox_inches='tight')
                logger.info(f"Visualization saved to {save_path}")
            
            # Convert matplotlib figure to numpy array
            fig.canvas.draw()
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            
            plt.close(fig)
            return img
            
        except Exception as e:
            logger.error(f"Visualization creation failed: {e}")
            return image
    
    def run_continuous(self, interval: int = 10, save_results: bool = True, 
                      display_gui: bool = False):
        """Run continuous crowd counting every interval seconds"""
        logger.info(f"Starting continuous crowd counting (interval: {interval}s)")
        
        # Initialize camera
        if not self._init_camera():
            logger.error("Failed to initialize camera")
            return
        
        # Create results directory
        results_dir = Path("crowd_counting_results")
        results_dir.mkdir(exist_ok=True)
        
        try:
            while True:
                loop_start = time.time()
                
                # Capture frame
                frame = self._capture_frame()
                if frame is None:
                    logger.warning("Failed to capture frame")
                    time.sleep(1)
                    continue
                
                # Perform inference
                count, density = self._count_people(frame)
                
                # Record system stats
                self.perf_monitor.record_system_stats()
                
                # Get performance stats
                stats = self.perf_monitor.get_stats()
                
                # Create timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # Log results
                logger.info(f"Count: {count:.1f}, Inference: {stats['avg_inference_time']:.3f}s, "
                          f"CPU: {stats['avg_cpu_usage']:.1f}%, RAM: {stats['avg_memory_usage']:.1f}%")
                
                # Create and save visualization
                if save_results:
                    save_path = results_dir / f"crowd_count_{timestamp}.png"
                    visualization = self._create_visualization(frame, count, density, str(save_path))
                    
                    # Also save raw data
                    np.savez(results_dir / f"data_{timestamp}.npz", 
                            image=frame, count=count, density=density)
                
                # Display GUI if requested
                if display_gui:
                    # This would need a proper GUI implementation
                    pass
                
                # Wait for next interval
                elapsed = time.time() - loop_start
                sleep_time = max(0, interval - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
        except KeyboardInterrupt:
            logger.info("Stopping crowd counting...")
        finally:
            self._cleanup()
    
    def _cleanup(self):
        """Clean up resources"""
        if self.camera:
            self.camera.stop()
            self.camera.close()
        if self.cap:
            self.cap.release()
        logger.info("Resources cleaned up")

def main():
    parser = argparse.ArgumentParser(description="Raspberry Pi 5 Crowd Counting System")
    parser.add_argument('--model', default='CSRNet', choices=['CSRNet', 'Bay', 'DM-Count', 'SFANet'],
                       help='Model to use for crowd counting')
    parser.add_argument('--weights', default='SHB', choices=['SHA', 'SHB', 'QNRF'],
                       help='Model weights (SHB for sparse crowds, SHA/QNRF for dense)')
    parser.add_argument('--interval', type=int, default=10,
                       help='Interval between captures in seconds')
    parser.add_argument('--image-size', nargs=2, type=int, default=[224, 224],
                       help='Image size for inference')
    parser.add_argument('--no-save', action='store_true',
                       help='Don\'t save results to disk')
    parser.add_argument('--gui', action='store_true',
                       help='Display GUI (requires display)')
    
    args = parser.parse_args()
    
    # Create crowd counter
    counter = CrowdCounter(
        model_name=args.model,
        model_weights=args.weights,
        image_size=tuple(args.image_size)
    )
    
    # Run continuous monitoring
    counter.run_continuous(
        interval=args.interval,
        save_results=not args.no_save,
        display_gui=args.gui
    )

if __name__ == "__main__":
    main() 