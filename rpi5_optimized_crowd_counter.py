#!/usr/bin/env python3
"""
Raspberry Pi 5 Optimized Crowd Counting System
Real-time crowd counting with LWCC using Pi Camera
Fully optimized for Pi 5 hardware and Raspberry Pi OS Bookworm
"""

import cv2
import numpy as np
import torch
import time
import threading
import os
import psutil
import platform
import subprocess
from pathlib import Path
from datetime import datetime
import logging
from typing import Tuple, Optional
import argparse
import sys

# Raspberry Pi specific imports with fallbacks
PI_CAMERA_AVAILABLE = False
CAMERA_TYPE = "none"

# Try Pi Camera 2 (libcamera)
try:
    from picamera2 import Picamera2
    PI_CAMERA_AVAILABLE = True
    CAMERA_TYPE = "picamera2"
    print("✅ Pi Camera 2 (libcamera) available")
except ImportError:
    try:
        # Fallback to legacy Pi Camera
        import picamera
        PI_CAMERA_AVAILABLE = True
        CAMERA_TYPE = "picamera"
        print("✅ Legacy Pi Camera available")
    except ImportError:
        print("⚠️  Pi Camera not available. Using OpenCV fallback.")

# LWCC patching for local cache
import shutil
import os.path as osp

# Pi 5 Specific Performance Optimizations
def setup_pi5_optimizations():
    """Configure Pi 5 specific optimizations"""
    
    # Check if we're actually on Pi 5
    try:
        with open('/proc/device-tree/model', 'r') as f:
            model = f.read()
        if 'Raspberry Pi 5' not in model:
            print(f"⚠️  Not running on Pi 5: {model.strip()}")
    except:
        print("⚠️  Could not detect Pi model")
    
    # ARM64 optimizations for Pi 5
    os.environ['DNNL_DEFAULT_FPMATH_MODE'] = 'BF16'  # BFloat16 for faster inference
    os.environ['LRU_CACHE_CAPACITY'] = '1024'  # Cache optimization
    os.environ['THP_MEM_ALLOC_ENABLE'] = '1'  # Transparent huge pages
    os.environ['OMP_NUM_THREADS'] = '4'  # Pi 5 has 4 cores
    
    # Pi 5 specific optimizations
    os.environ['PYTORCH_THREAD_LOCAL_MEMORY_POOL'] = '1'
    os.environ['PYTORCH_QNNPACK_RUNTIME_QUANTIZATION'] = '1'
    
    # Memory optimizations for Pi 5
    os.environ['MALLOC_TRIM_THRESHOLD_'] = '100000'
    os.environ['MALLOC_MMAP_THRESHOLD_'] = '100000'
    
    # CPU governor optimization
    try:
        subprocess.run(['sudo', 'sh', '-c', 
                       'echo performance > /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor'],
                      check=False, capture_output=True)
    except:
        pass  # Not critical if this fails

# Apply optimizations at import time
setup_pi5_optimizations()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Pi5PerformanceMonitor:
    """Enhanced performance monitoring for Pi 5"""
    
    def __init__(self):
        self.inference_times = []
        self.memory_usage = []
        self.cpu_usage = []
        self.temperature_readings = []
        self.throttle_events = []
    
    def record_inference_time(self, inference_time: float):
        self.inference_times.append(inference_time)
        if len(self.inference_times) > 100:
            self.inference_times.pop(0)
    
    def record_system_stats(self):
        # Standard system stats
        memory = psutil.virtual_memory()
        cpu = psutil.cpu_percent()
        
        self.memory_usage.append(memory.percent)
        self.cpu_usage.append(cpu)
        
        # Pi 5 specific monitoring
        try:
            # Temperature monitoring
            temp_result = subprocess.run(['vcgencmd', 'measure_temp'], 
                                       capture_output=True, text=True, timeout=2)
            if temp_result.returncode == 0:
                temp_str = temp_result.stdout.strip()
                temp = float(temp_str.replace('temp=', '').replace("'C", ''))
                self.temperature_readings.append(temp)
            
            # Throttling detection
            throttle_result = subprocess.run(['vcgencmd', 'get_throttled'], 
                                           capture_output=True, text=True, timeout=2)
            if throttle_result.returncode == 0:
                throttle_hex = throttle_result.stdout.strip().replace('throttled=', '')
                throttle_int = int(throttle_hex, 16)
                self.throttle_events.append(throttle_int)
                
        except Exception as e:
            logger.debug(f"Pi monitoring failed: {e}")
        
        # Keep only last 100 readings
        for attr in ['memory_usage', 'cpu_usage', 'temperature_readings', 'throttle_events']:
            readings = getattr(self, attr)
            if len(readings) > 100:
                readings.pop(0)
    
    def get_stats(self) -> dict:
        stats = {
            'avg_inference_time': np.mean(self.inference_times) if self.inference_times else 0,
            'avg_memory_usage': np.mean(self.memory_usage) if self.memory_usage else 0,
            'avg_cpu_usage': np.mean(self.cpu_usage) if self.cpu_usage else 0,
            'fps': 1.0 / np.mean(self.inference_times) if self.inference_times else 0
        }
        
        # Pi 5 specific stats
        if self.temperature_readings:
            stats['avg_temperature'] = np.mean(self.temperature_readings)
            stats['max_temperature'] = np.max(self.temperature_readings)
        
        if self.throttle_events:
            stats['throttle_status'] = self.throttle_events[-1]
            stats['throttle_warning'] = any(x != 0 for x in self.throttle_events[-10:])
        
        return stats

class Pi5Camera:
    """Pi 5 optimized camera interface"""
    
    def __init__(self, image_size: Tuple[int, int] = (224, 224)):
        self.image_size = image_size
        self.camera = None
        self.cap = None
        self.camera_type = CAMERA_TYPE
        
    def initialize(self) -> bool:
        """Initialize camera with Pi 5 optimizations"""
        
        if self.camera_type == "picamera2":
            return self._init_picamera2()
        elif self.camera_type == "picamera":
            return self._init_legacy_picamera()
        else:
            return self._init_opencv_camera()
    
    def _init_picamera2(self) -> bool:
        """Initialize Pi Camera 2 with libcamera (preferred for Pi 5)"""
        try:
            self.camera = Picamera2()
            
            # Optimized configuration for Pi 5
            config = self.camera.create_preview_configuration(
                main={
                    "format": "RGB888", 
                    "size": self.image_size
                },
                controls={
                    "FrameRate": 30,
                    "ExposureTime": 10000,  # 10ms exposure
                    "AnalogueGain": 1.0
                }
            )
            
            self.camera.configure(config)
            self.camera.start()
            
            # Warm-up capture
            time.sleep(1)
            _ = self.camera.capture_array()
            
            logger.info("Pi Camera 2 (libcamera) initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Pi Camera 2 initialization failed: {e}")
            return False
    
    def _init_legacy_picamera(self) -> bool:
        """Initialize legacy Pi Camera"""
        try:
            import picamera
            self.camera = picamera.PiCamera()
            self.camera.resolution = self.image_size
            self.camera.framerate = 30
            
            # Warm-up
            time.sleep(2)
            
            logger.info("Legacy Pi Camera initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Legacy Pi Camera initialization failed: {e}")
            return False
    
    def _init_opencv_camera(self) -> bool:
        """Initialize USB/OpenCV camera with Pi 5 optimizations"""
        try:
            # Try different camera indices for USB cameras
            for camera_idx in [0, 1, 2]:
                self.cap = cv2.VideoCapture(camera_idx)
                if self.cap.isOpened():
                    break
                self.cap.release()
                self.cap = None
            
            if self.cap is None:
                return False
            
            # Optimize for Pi 5
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.image_size[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.image_size[1])
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency
            
            # Test capture
            ret, _ = self.cap.read()
            if not ret:
                self.cap.release()
                return False
            
            logger.info("OpenCV camera initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"OpenCV camera initialization failed: {e}")
            return False
    
    def capture_frame(self) -> Optional[np.ndarray]:
        """Capture frame with error handling"""
        try:
            if self.camera_type == "picamera2" and self.camera:
                frame = self.camera.capture_array()
                return frame
                
            elif self.camera_type == "picamera" and self.camera:
                import picamera
                import io
                stream = io.BytesIO()
                self.camera.capture(stream, format='rgb')
                stream.seek(0)
                frame = np.frombuffer(stream.getvalue(), dtype=np.uint8)
                frame = frame.reshape((self.image_size[1], self.image_size[0], 3))
                return frame
                
            elif self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    # Convert BGR to RGB and resize
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = cv2.resize(frame, self.image_size)
                    return frame
            
            return None
            
        except Exception as e:
            logger.error(f"Frame capture failed: {e}")
            return None
    
    def cleanup(self):
        """Clean up camera resources"""
        try:
            if self.camera_type == "picamera2" and self.camera:
                self.camera.stop()
                self.camera.close()
            elif self.camera_type == "picamera" and self.camera:
                self.camera.close()
            elif self.cap:
                self.cap.release()
        except Exception as e:
            logger.debug(f"Camera cleanup error: {e}")

class Pi5CrowdCounter:
    """Pi 5 optimized crowd counting system"""
    
    def __init__(self, model_name: str = "CSRNet", model_weights: str = "SHB", 
                 image_size: Tuple[int, int] = (224, 224)):
        
        self.model_name = model_name
        self.model_weights = model_weights
        self.image_size = image_size
        
        # Pi 5 performance monitoring
        self.perf_monitor = Pi5PerformanceMonitor()
        
        # Pi 5 optimized camera
        self.camera = Pi5Camera(image_size)
        
        # Setup LWCC with local cache
        self._setup_lwcc_patches()
        
        # Load and optimize model for Pi 5
        self._load_model()
        
        logger.info(f"Pi 5 Crowd counter initialized with {model_name}-{model_weights}")
    
    def _setup_lwcc_patches(self):
        """Apply LWCC patches for local cache (same as before)"""
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
        """Load and optimize model for Pi 5"""
        logger.info("Loading LWCC model with Pi 5 optimizations...")
        
        # Pi 5 specific PyTorch settings
        torch.backends.quantized.engine = 'qnnpack'  # ARM-optimized
        torch.set_num_threads(4)  # Pi 5 has 4 cores
        torch.set_grad_enabled(False)  # Inference only
        
        # Enable Pi 5 optimizations
        if hasattr(torch.backends, 'mkldnn'):
            torch.backends.mkldnn.enabled = True
        
        # Import LWCC after patches
        from lwcc import LWCC
        self.lwcc = LWCC
        
        # Pre-load and cache model
        try:
            dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            start_time = time.time()
            _ = self.lwcc.get_count(dummy_image, model_name=self.model_name, 
                                   model_weights=self.model_weights, return_density=True)
            load_time = time.time() - start_time
            logger.info(f"Model loaded and cached successfully in {load_time:.2f}s")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def _count_people(self, image: np.ndarray) -> Tuple[float, np.ndarray]:
        """Perform optimized crowd counting inference"""
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
            return 0.0, np.zeros((56, 56))
    
    def _create_visualization(self, image: np.ndarray, count: float, 
                            density: np.ndarray, save_path: str = None) -> Optional[np.ndarray]:
        """Create visualization optimized for Pi 5"""
        try:
            # Use matplotlib Agg backend for headless operation
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            # Create density heatmap
            if density.max() > 0:
                normalized_density = (density / density.max() * 255).astype(np.uint8)
            else:
                normalized_density = np.zeros_like(density, dtype=np.uint8)
            
            heat = cv2.applyColorMap(normalized_density, cv2.COLORMAP_JET)
            heat = cv2.resize(heat, self.image_size, interpolation=cv2.INTER_CUBIC)
            
            # Create figure with specific DPI for Pi 5
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), dpi=80)
            
            ax1.imshow(image)
            ax1.set_title(f"Count: {count:.1f}")
            ax1.axis('off')
            
            ax2.imshow(heat)
            ax2.set_title("Density Map")
            ax2.axis('off')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=80, bbox_inches='tight')
                logger.info(f"Visualization saved: {save_path}")
            
            plt.close(fig)
            return heat
            
        except Exception as e:
            logger.error(f"Visualization failed: {e}")
            return None
    
    def run_continuous(self, interval: int = 10, save_results: bool = True):
        """Run continuous crowd counting optimized for Pi 5"""
        logger.info(f"Starting Pi 5 crowd counting (interval: {interval}s)")
        
        # Initialize camera
        if not self.camera.initialize():
            logger.error("Failed to initialize camera")
            return
        
        # Create results directory
        results_dir = Path("crowd_counting_results")
        results_dir.mkdir(exist_ok=True)
        
        try:
            while True:
                loop_start = time.time()
                
                # Capture frame
                frame = self.camera.capture_frame()
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
                
                # Enhanced logging with Pi 5 specific info
                log_msg = f"Count: {count:.1f}, Inference: {stats['avg_inference_time']:.3f}s, "
                log_msg += f"CPU: {stats['avg_cpu_usage']:.1f}%, RAM: {stats['avg_memory_usage']:.1f}%"
                
                if 'avg_temperature' in stats:
                    log_msg += f", Temp: {stats['avg_temperature']:.1f}°C"
                
                if stats.get('throttle_warning', False):
                    log_msg += " ⚠️ THROTTLING"
                
                logger.info(log_msg)
                
                # Save results
                if save_results:
                    save_path = results_dir / f"crowd_count_{timestamp}.png"
                    self._create_visualization(frame, count, density, str(save_path))
                    
                    # Save raw data
                    np.savez(results_dir / f"data_{timestamp}.npz", 
                            image=frame, count=count, density=density, stats=stats)
                
                # Wait for next interval
                elapsed = time.time() - loop_start
                sleep_time = max(0, interval - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
        except KeyboardInterrupt:
            logger.info("Stopping crowd counting...")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        self.camera.cleanup()
        logger.info("Pi 5 crowd counter cleaned up")

def main():
    """Main function with Pi 5 specific argument parsing"""
    parser = argparse.ArgumentParser(description="Raspberry Pi 5 Optimized Crowd Counting System")
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
    
    args = parser.parse_args()
    
    # System info
    logger.info(f"Python: {sys.version}")
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"Camera: {CAMERA_TYPE}")
    
    # Create Pi 5 optimized crowd counter
    counter = Pi5CrowdCounter(
        model_name=args.model,
        model_weights=args.weights,
        image_size=tuple(args.image_size)
    )
    
    # Run continuous monitoring
    counter.run_continuous(
        interval=args.interval,
        save_results=not args.no_save
    )

if __name__ == "__main__":
    main() 