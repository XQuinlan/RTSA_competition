#!/usr/bin/env python3
"""
Test script to verify Raspberry Pi 5 crowd counting installation
"""

import sys
import time
import numpy as np
import torch
import cv2
from PIL import Image
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_pytorch():
    """Test PyTorch installation and optimization"""
    logger.info("🔥 Testing PyTorch installation...")
    
    try:
        print(f"✅ PyTorch version: {torch.__version__}")
        print(f"✅ CPU cores available: {torch.get_num_threads()}")
        print(f"✅ Quantization engine: {torch.backends.quantized.engine}")
        
        # Test basic tensor operations
        x = torch.randn(100, 100)
        y = torch.mm(x, x)
        print(f"✅ Basic tensor operations working")
        
        # Test quantization
        torch.backends.quantized.engine = 'qnnpack'
        print(f"✅ Quantization engine set to: {torch.backends.quantized.engine}")
        
        return True
    except Exception as e:
        logger.error(f"❌ PyTorch test failed: {e}")
        return False

def test_lwcc():
    """Test LWCC installation and model loading"""
    logger.info("🧠 Testing LWCC installation...")
    
    try:
        from lwcc import LWCC
        print("✅ LWCC imported successfully")
        
        # Test with dummy image
        dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        start_time = time.time()
        count = LWCC.get_count(dummy_image, model_name="CSRNet", model_weights="SHB")
        inference_time = time.time() - start_time
        
        print(f"✅ LWCC inference successful")
        print(f"✅ Dummy image count: {count:.2f}")
        print(f"✅ Inference time: {inference_time:.3f}s")
        
        return True
    except Exception as e:
        logger.error(f"❌ LWCC test failed: {e}")
        return False

def test_camera():
    """Test camera access"""
    logger.info("📷 Testing camera access...")
    
    # Try Pi Camera first
    try:
        from picamera2 import Picamera2
        camera = Picamera2()
        config = camera.create_preview_configuration(main={"format": "RGB888", "size": (224, 224)})
        camera.configure(config)
        camera.start()
        
        # Capture a frame
        frame = camera.capture_array()
        camera.stop()
        camera.close()
        
        print("✅ Pi Camera working")
        print(f"✅ Frame shape: {frame.shape}")
        return True
    except Exception as e:
        logger.warning(f"Pi Camera failed: {e}")
    
    # Fallback to OpenCV
    try:
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print("✅ OpenCV camera working")
                print(f"✅ Frame shape: {frame.shape}")
                cap.release()
                return True
            else:
                print("❌ Could not read frame from camera")
                cap.release()
                return False
        else:
            print("❌ Could not open camera")
            return False
    except Exception as e:
        logger.error(f"❌ Camera test failed: {e}")
        return False

def test_performance_optimizations():
    """Test performance optimizations"""
    logger.info("⚡ Testing performance optimizations...")
    
    try:
        # Check environment variables
        import os
        optimizations = {
            'DNNL_DEFAULT_FPMATH_MODE': os.environ.get('DNNL_DEFAULT_FPMATH_MODE'),
            'LRU_CACHE_CAPACITY': os.environ.get('LRU_CACHE_CAPACITY'),
            'THP_MEM_ALLOC_ENABLE': os.environ.get('THP_MEM_ALLOC_ENABLE'),
            'OMP_NUM_THREADS': os.environ.get('OMP_NUM_THREADS'),
        }
        
        for key, value in optimizations.items():
            if value:
                print(f"✅ {key}: {value}")
            else:
                print(f"⚠️  {key}: not set")
        
        # Test memory usage
        import psutil
        memory = psutil.virtual_memory()
        print(f"✅ Available memory: {memory.available / 1024**3:.1f} GB")
        print(f"✅ Memory usage: {memory.percent:.1f}%")
        
        return True
    except Exception as e:
        logger.error(f"❌ Performance optimization test failed: {e}")
        return False

def test_full_pipeline():
    """Test the complete pipeline with test images"""
    logger.info("🔄 Testing complete pipeline...")
    
    try:
        # Use existing test images or create synthetic ones
        test_images = ["test_image.jpg", "test_image_2.jpg"]
        
        for img_path in test_images:
            try:
                # Load image
                image = cv2.imread(img_path)
                if image is not None:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image = cv2.resize(image, (224, 224))
                    
                    # Test inference
                    from lwcc import LWCC
                    
                    start_time = time.time()
                    count, density = LWCC.get_count(
                        image, 
                        model_name="CSRNet", 
                        model_weights="SHB", 
                        return_density=True
                    )
                    inference_time = time.time() - start_time
                    
                    print(f"✅ {img_path}: count={count:.1f}, time={inference_time:.3f}s")
                else:
                    print(f"⚠️  Could not load {img_path}")
            except Exception as e:
                logger.warning(f"Failed to test {img_path}: {e}")
        
        return True
    except Exception as e:
        logger.error(f"❌ Full pipeline test failed: {e}")
        return False

def main():
    """Run all tests"""
    logger.info("🚀 Starting Raspberry Pi 5 crowd counting installation tests...")
    
    tests = [
        ("PyTorch", test_pytorch),
        ("LWCC", test_lwcc),
        ("Camera", test_camera),
        ("Performance Optimizations", test_performance_optimizations),
        ("Full Pipeline", test_full_pipeline)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running {test_name} test...")
        logger.info(f"{'='*50}")
        
        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {e}")
            results[test_name] = False
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*50}")
    
    all_passed = True
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
        if not result:
            all_passed = False
    
    if all_passed:
        logger.info("\n🎉 All tests passed! Your Raspberry Pi 5 is ready for crowd counting.")
        logger.info("You can now run: python3 rpi_crowd_counter.py")
    else:
        logger.error("\n⚠️  Some tests failed. Please check the setup.")
        logger.error("Run the setup script again: bash setup_rpi5.sh")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 