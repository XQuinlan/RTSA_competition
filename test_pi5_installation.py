#!/usr/bin/env python3
"""
Comprehensive Raspberry Pi 5 Installation Test Suite
Tests all Pi 5 specific hardware and software components
"""

import sys
import time
import numpy as np
import torch
import cv2
import platform
import subprocess
import os
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_pi5_hardware():
    """Test Pi 5 specific hardware detection"""
    logger.info("🔧 Testing Pi 5 hardware detection...")
    
    try:
        # Check Pi model
        with open('/proc/device-tree/model', 'r') as f:
            model = f.read().strip()
        
        if 'Raspberry Pi 5' in model:
            print(f"✅ Detected: {model}")
        else:
            print(f"⚠️  Not Pi 5: {model}")
            return False
        
        # Check architecture
        arch = platform.machine()
        if arch == 'aarch64':
            print(f"✅ Architecture: {arch} (64-bit ARM)")
        else:
            print(f"❌ Architecture: {arch} (requires aarch64)")
            return False
        
        # Check CPU info
        with open('/proc/cpuinfo', 'r') as f:
            cpuinfo = f.read()
        
        if 'BCM2712' in cpuinfo:  # Pi 5 SoC
            print("✅ Pi 5 BCM2712 SoC detected")
        else:
            print("⚠️  Pi 5 SoC not detected")
        
        # Check CPU cores
        cpu_count = os.cpu_count()
        print(f"✅ CPU cores: {cpu_count}")
        
        # Check memory
        with open('/proc/meminfo', 'r') as f:
            for line in f:
                if line.startswith('MemTotal'):
                    mem_kb = int(line.split()[1])
                    mem_gb = mem_kb / 1024 / 1024
                    print(f"✅ Total RAM: {mem_gb:.1f} GB")
                    break
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Hardware detection failed: {e}")
        return False

def test_pi5_performance_features():
    """Test Pi 5 performance features and optimizations"""
    logger.info("⚡ Testing Pi 5 performance features...")
    
    try:
        # Test CPU governor
        try:
            with open('/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor', 'r') as f:
                governor = f.read().strip()
            print(f"✅ CPU governor: {governor}")
        except:
            print("⚠️  Could not read CPU governor")
        
        # Test temperature monitoring
        try:
            result = subprocess.run(['vcgencmd', 'measure_temp'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                temp = result.stdout.strip()
                print(f"✅ Temperature: {temp}")
            else:
                print("⚠️  Temperature monitoring unavailable")
        except:
            print("⚠️  vcgencmd not available")
        
        # Test throttling status
        try:
            result = subprocess.run(['vcgencmd', 'get_throttled'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                throttled = result.stdout.strip()
                throttle_val = int(throttled.replace('throttled=', ''), 16)
                if throttle_val == 0:
                    print("✅ No throttling detected")
                else:
                    print(f"⚠️  Throttling detected: {throttled}")
        except:
            print("⚠️  Throttling check unavailable")
        
        # Test environment variables
        env_vars = {
            'DNNL_DEFAULT_FPMATH_MODE': 'BF16 optimization',
            'LRU_CACHE_CAPACITY': 'Cache optimization',
            'THP_MEM_ALLOC_ENABLE': 'Huge pages',
            'OMP_NUM_THREADS': 'Thread optimization'
        }
        
        for var, desc in env_vars.items():
            value = os.environ.get(var)
            if value:
                print(f"✅ {desc}: {var}={value}")
            else:
                print(f"⚠️  {desc}: {var} not set")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Performance feature test failed: {e}")
        return False

def test_pytorch_pi5_optimizations():
    """Test PyTorch optimizations for Pi 5"""
    logger.info("🔥 Testing PyTorch Pi 5 optimizations...")
    
    try:
        print(f"✅ PyTorch version: {torch.__version__}")
        print(f"✅ CPU threads: {torch.get_num_threads()}")
        
        # Test quantization engine
        if hasattr(torch.backends, 'quantized'):
            print(f"✅ Quantization engine: {torch.backends.quantized.engine}")
            
            # Set to QNNPACK for ARM
            torch.backends.quantized.engine = 'qnnpack'
            print("✅ Set quantization engine to qnnpack")
        
        # Test basic operations with timing
        print("Testing tensor operations...")
        
        # Small tensor test
        start_time = time.time()
        x = torch.randn(100, 100)
        y = torch.mm(x, x)
        small_time = time.time() - start_time
        print(f"✅ Small tensor ops: {small_time:.4f}s")
        
        # Larger tensor test (more realistic for inference)
        start_time = time.time()
        x = torch.randn(512, 512)
        y = torch.mm(x, x)
        large_time = time.time() - start_time
        print(f"✅ Large tensor ops: {large_time:.4f}s")
        
        # Test memory efficiency
        torch.set_grad_enabled(False)  # Inference mode
        print("✅ Gradient computation disabled")
        
        # Test threading
        original_threads = torch.get_num_threads()
        torch.set_num_threads(4)  # Pi 5 optimal
        if torch.get_num_threads() == 4:
            print("✅ Thread count set to 4")
        torch.set_num_threads(original_threads)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ PyTorch optimization test failed: {e}")
        return False

def test_camera_pi5():
    """Test camera functionality on Pi 5"""
    logger.info("📷 Testing Pi 5 camera functionality...")
    
    camera_tested = False
    
    # Test Pi Camera 2 (libcamera)
    try:
        from picamera2 import Picamera2
        camera = Picamera2()
        
        config = camera.create_preview_configuration(
            main={"format": "RGB888", "size": (224, 224)}
        )
        camera.configure(config)
        camera.start()
        
        # Test capture
        frame = camera.capture_array()
        camera.stop()
        camera.close()
        
        print("✅ Pi Camera 2 (libcamera) working")
        print(f"✅ Frame shape: {frame.shape}")
        camera_tested = True
        
    except ImportError:
        print("⚠️  picamera2 not available")
    except Exception as e:
        print(f"⚠️  Pi Camera 2 failed: {e}")
    
    # Test legacy Pi Camera
    if not camera_tested:
        try:
            import picamera
            camera = picamera.PiCamera()
            camera.resolution = (224, 224)
            time.sleep(1)
            camera.close()
            
            print("✅ Legacy Pi Camera working")
            camera_tested = True
            
        except ImportError:
            print("⚠️  Legacy picamera not available")
        except Exception as e:
            print(f"⚠️  Legacy Pi Camera failed: {e}")
    
    # Test USB/OpenCV camera
    if not camera_tested:
        try:
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    print("✅ USB/OpenCV camera working")
                    print(f"✅ Frame shape: {frame.shape}")
                    camera_tested = True
                cap.release()
        except Exception as e:
            print(f"⚠️  USB camera test failed: {e}")
    
    if not camera_tested:
        print("❌ No camera detected")
        return False
    
    return True

def test_lwcc_pi5():
    """Test LWCC library on Pi 5"""
    logger.info("🧠 Testing LWCC on Pi 5...")
    
    try:
        from lwcc import LWCC
        print("✅ LWCC imported successfully")
        
        # Test with dummy image
        dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        # Test different models and measure performance
        models_to_test = [
            ("CSRNet", "SHB"),
            ("CSRNet", "SHA"),
        ]
        
        for model_name, model_weights in models_to_test:
            try:
                start_time = time.time()
                count = LWCC.get_count(dummy_image, 
                                     model_name=model_name, 
                                     model_weights=model_weights)
                inference_time = time.time() - start_time
                
                print(f"✅ {model_name}-{model_weights}: count={count:.2f}, time={inference_time:.3f}s")
                
                # Test with density map
                start_time = time.time()
                count, density = LWCC.get_count(dummy_image, 
                                               model_name=model_name, 
                                               model_weights=model_weights,
                                               return_density=True)
                density_time = time.time() - start_time
                
                print(f"✅ {model_name}-{model_weights} with density: time={density_time:.3f}s, density shape={density.shape}")
                
            except Exception as e:
                print(f"⚠️  {model_name}-{model_weights} failed: {e}")
        
        return True
        
    except ImportError as e:
        logger.error(f"❌ LWCC import failed: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ LWCC test failed: {e}")
        return False

def test_full_pipeline_pi5():
    """Test complete pipeline on Pi 5"""
    logger.info("🔄 Testing complete Pi 5 pipeline...")
    
    try:
        # Import the Pi 5 optimized crowd counter
        sys.path.append('.')
        
        # Create a test image
        test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        # Test the main components
        from lwcc import LWCC
        
        # Time the complete pipeline
        start_time = time.time()
        
        # Load model
        torch.backends.quantized.engine = 'qnnpack'
        torch.set_num_threads(4)
        
        # Inference
        count, density = LWCC.get_count(test_image, 
                                       model_name="CSRNet", 
                                       model_weights="SHB",
                                       return_density=True)
        
        # Visualization
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        normalized_density = (density / density.max() * 255).astype(np.uint8) if density.max() > 0 else density
        heat = cv2.applyColorMap(normalized_density, cv2.COLORMAP_JET)
        
        total_time = time.time() - start_time
        
        print(f"✅ Complete pipeline: {total_time:.3f}s")
        print(f"✅ Count: {count:.2f}")
        print(f"✅ Density shape: {density.shape}")
        print(f"✅ Heatmap shape: {heat.shape}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Full pipeline test failed: {e}")
        return False

def test_system_resources():
    """Test system resource availability"""
    logger.info("💾 Testing system resources...")
    
    try:
        import psutil
        
        # Memory test
        memory = psutil.virtual_memory()
        print(f"✅ Available memory: {memory.available / 1024**3:.1f} GB")
        print(f"✅ Memory usage: {memory.percent:.1f}%")
        
        if memory.available < 2 * 1024**3:  # Less than 2GB available
            print("⚠️  Low memory available")
        
        # CPU test
        cpu_percent = psutil.cpu_percent(interval=1)
        print(f"✅ CPU usage: {cpu_percent:.1f}%")
        
        # Disk test
        disk = psutil.disk_usage('/')
        print(f"✅ Disk free: {disk.free / 1024**3:.1f} GB")
        
        if disk.free < 5 * 1024**3:  # Less than 5GB free
            print("⚠️  Low disk space")
        
        # Swap test
        swap = psutil.swap_memory()
        if swap.total > 0:
            print(f"✅ Swap available: {swap.total / 1024**3:.1f} GB")
        else:
            print("⚠️  No swap configured")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ System resource test failed: {e}")
        return False

def main():
    """Run comprehensive Pi 5 test suite"""
    logger.info("🚀 Starting Raspberry Pi 5 comprehensive test suite...")
    
    tests = [
        ("Pi 5 Hardware Detection", test_pi5_hardware),
        ("Pi 5 Performance Features", test_pi5_performance_features),
        ("PyTorch Pi 5 Optimizations", test_pytorch_pi5_optimizations),
        ("Pi 5 Camera", test_camera_pi5),
        ("LWCC on Pi 5", test_lwcc_pi5),
        ("System Resources", test_system_resources),
        ("Complete Pipeline", test_full_pipeline_pi5)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running {test_name} test...")
        logger.info(f"{'='*60}")
        
        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {e}")
            results[test_name] = False
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("RASPBERRY PI 5 TEST SUMMARY")
    logger.info(f"{'='*60}")
    
    all_passed = True
    critical_failed = False
    
    critical_tests = ["Pi 5 Hardware Detection", "PyTorch Pi 5 Optimizations", "LWCC on Pi 5"]
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        critical = "🔥 CRITICAL" if test_name in critical_tests else ""
        logger.info(f"{test_name}: {status} {critical}")
        
        if not result:
            all_passed = False
            if test_name in critical_tests:
                critical_failed = True
    
    if all_passed:
        logger.info("\n🎉 All tests passed! Your Pi 5 is fully optimized for crowd counting.")
        logger.info("✅ Ready to run: python3 rpi5_optimized_crowd_counter.py")
    elif critical_failed:
        logger.error("\n❌ Critical tests failed. Please check your Pi 5 setup.")
        logger.error("Run: ./setup_rpi5_fixed.sh")
    else:
        logger.warning("\n⚠️  Some non-critical tests failed, but system should work.")
        logger.info("You can try running: python3 rpi5_optimized_crowd_counter.py")
    
    # Pi 5 specific recommendations
    logger.info(f"\n{'='*60}")
    logger.info("RASPBERRY PI 5 RECOMMENDATIONS")
    logger.info(f"{'='*60}")
    
    recommendations = [
        "🔧 Use heatsinks and active cooling for sustained performance",
        "⚡ Ensure 5V 3A official power supply",
        "💾 Use fast microSD (A2 rated) or SSD for better I/O",
        "🌡️  Monitor temperature with: watch vcgencmd measure_temp",
        "🔄 Check for throttling with: vcgencmd get_throttled",
        "📊 Monitor with: htop, free -h, df -h"
    ]
    
    for rec in recommendations:
        logger.info(rec)
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 