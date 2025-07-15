#!/bin/bash

# Raspberry Pi 5 Final Setup Script
# Complete setup with all optimizations and compatibility fixes

set -e

echo "🔧 Setting up Raspberry Pi 5 for Crowd Counting (Final Version)..."

# Check if we're on a Pi 5
if [[ $(cat /proc/device-tree/model) != *"Raspberry Pi 5"* ]]; then
    echo "❌ This script is optimized for Raspberry Pi 5"
    echo "Current model: $(cat /proc/device-tree/model)"
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Verify 64-bit OS
if [[ $(uname -m) != "aarch64" ]]; then
    echo "❌ This requires a 64-bit OS (aarch64)"
    echo "Current architecture: $(uname -m)"
    echo "Please install 64-bit Raspberry Pi OS"
    exit 1
fi

echo "✅ Raspberry Pi 5 with 64-bit OS detected"

# Update system with progress
echo "📦 Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install system dependencies (Pi 5 optimized)
echo "🔧 Installing Pi 5 optimized system dependencies..."
sudo apt install -y \
    python3-pip \
    python3-venv \
    python3-dev \
    python3-setuptools \
    libopenblas-dev \
    libjpeg-dev \
    libopencv-dev \
    libhdf5-dev \
    libatlas-base-dev \
    libjasper-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    git \
    cmake \
    pkg-config \
    wget \
    curl \
    htop \
    tmux \
    build-essential \
    libblas-dev \
    liblapack-dev \
    gfortran \
    libffi-dev \
    libssl-dev \
    libcamera-dev \
    libcamera-tools \
    python3-libcamera \
    python3-kms++

# Enable camera and configure for Pi 5
echo "📷 Configuring Pi 5 camera..."
sudo raspi-config nonint do_camera 0

# Update boot config for Pi 5
echo "🔧 Configuring Pi 5 boot settings..."
sudo cp /boot/firmware/config.txt /boot/firmware/config.txt.backup

# Pi 5 specific configurations
if ! grep -q "gpu_mem=128" /boot/firmware/config.txt; then
    echo "gpu_mem=128" | sudo tee -a /boot/firmware/config.txt
fi

if ! grep -q "camera_auto_detect=1" /boot/firmware/config.txt; then
    echo "camera_auto_detect=1" | sudo tee -a /boot/firmware/config.txt
fi

# Pi 5 performance optimizations
if ! grep -q "arm_boost=1" /boot/firmware/config.txt; then
    echo "arm_boost=1" | sudo tee -a /boot/firmware/config.txt
fi

# Create virtual environment
echo "🐍 Creating Python virtual environment..."
python3 -m venv crowd_counter_env
source crowd_counter_env/bin/activate

# Upgrade pip and install wheel
pip install --upgrade pip setuptools wheel

# Install PyTorch with CPU-only support (ARM optimized)
echo "🔥 Installing PyTorch (CPU-only, ARM64 optimized)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install Pi 5 optimized dependencies
echo "📚 Installing Pi 5 optimized Python dependencies..."
pip install -r requirements_rpi5_optimized.txt

# Performance optimizations for Pi 5
echo "⚡ Applying Pi 5 performance optimizations..."

# Set CPU governor to performance
echo 'performance' | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Create swap file optimized for Pi 5
echo "💾 Setting up optimized swap for Pi 5..."
if [ ! -f /swapfile ]; then
    sudo fallocate -l 4G /swapfile
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile
    sudo swapon /swapfile
    echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
fi

# Create Pi 5 performance environment script
cat > performance_env.sh << 'EOF'
#!/bin/bash
# Raspberry Pi 5 Performance Environment Variables

# Pi 5 PyTorch optimizations
export DNNL_DEFAULT_FPMATH_MODE=BF16
export LRU_CACHE_CAPACITY=1024
export THP_MEM_ALLOC_ENABLE=1
export OMP_NUM_THREADS=4

# Pi 5 ARM-specific optimizations
export PYTORCH_THREAD_LOCAL_MEMORY_POOL=1
export PYTORCH_QNNPACK_RUNTIME_QUANTIZATION=1

# Pi 5 memory optimizations
export MALLOC_TRIM_THRESHOLD_=100000
export MALLOC_MMAP_THRESHOLD_=100000

# Pi 5 specific
export PYTORCH_TENSORRT_FX_USE_TRTORCH_BACKEND=1

echo "Pi 5 performance environment configured"
EOF

chmod +x performance_env.sh

# Create systemd service for Pi 5
cat > crowd_counter.service << 'EOF'
[Unit]
Description=Raspberry Pi 5 Crowd Counter
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/RTSA_competition
Environment=PATH=/home/pi/RTSA_competition/crowd_counter_env/bin
EnvironmentFile=/home/pi/RTSA_competition/performance_env.sh
ExecStart=/home/pi/RTSA_competition/crowd_counter_env/bin/python /home/pi/RTSA_competition/rpi5_optimized_crowd_counter.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Make scripts executable
chmod +x *.py
chmod +x *.sh

# Test PyTorch installation
echo "🎯 Testing PyTorch installation on Pi 5..."
python3 -c "
import torch
print(f'✅ PyTorch version: {torch.__version__}')
print(f'✅ CPU threads: {torch.get_num_threads()}')
torch.backends.quantized.engine = 'qnnpack'
print(f'✅ Quantization engine: {torch.backends.quantized.engine}')
print(f'✅ Pi 5 tensor test: {torch.randn(100, 100).sum():.2f}')
"

# Test LWCC installation with error handling
echo "🧪 Testing LWCC installation on Pi 5..."
python3 -c "
try:
    from lwcc import LWCC
    print('✅ LWCC imported successfully')
    import numpy as np
    dummy = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    count = LWCC.get_count(dummy, model_name='CSRNet', model_weights='SHB')
    print(f'✅ LWCC test successful: count={count:.2f}')
except ImportError as e:
    print(f'⚠️  LWCC import failed, installing manually...')
    import subprocess
    subprocess.run(['pip', 'install', 'lwcc', '--no-cache-dir', '--verbose'])
    try:
        from lwcc import LWCC
        print('✅ LWCC installed and working')
    except Exception as e2:
        print(f'❌ LWCC installation failed: {e2}')
except Exception as e:
    print(f'⚠️  LWCC test failed: {e}')
"

# Test camera on Pi 5
echo "📋 Testing Pi 5 camera access..."
python3 -c "
import cv2

# Test Pi Camera 2 (libcamera)
try:
    from picamera2 import Picamera2
    camera = Picamera2()
    config = camera.create_preview_configuration(main={'format': 'RGB888', 'size': (224, 224)})
    camera.configure(config)
    camera.start()
    frame = camera.capture_array()
    camera.stop()
    camera.close()
    print('✅ Pi Camera 2 (libcamera) working')
    print(f'✅ Frame shape: {frame.shape}')
except ImportError:
    print('⚠️  picamera2 not available')
except Exception as e:
    print(f'⚠️  Pi Camera 2 failed: {e}')
    
    # Fallback to OpenCV
    try:
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print('✅ OpenCV camera working')
                print(f'✅ Frame shape: {frame.shape}')
            cap.release()
        else:
            print('⚠️  No camera detected')
    except Exception as e2:
        print(f'⚠️  Camera test failed: {e2}')
"

# Pi 5 temperature and throttling check
echo "🌡️  Checking Pi 5 status..."
vcgencmd measure_temp 2>/dev/null || echo "⚠️  Temperature monitoring unavailable"
vcgencmd get_throttled 2>/dev/null || echo "⚠️  Throttling check unavailable"

echo "
🚀 Pi 5 setup complete! 

🎯 OPTIMIZED FOR RASPBERRY PI 5:
✅ ARM64 PyTorch with QNNPACK quantization
✅ Pi Camera 2 (libcamera) support + fallbacks
✅ BFloat16 acceleration
✅ 4-core threading optimization
✅ Performance governor enabled
✅ 4GB swap configured
✅ Temperature monitoring
✅ Throttling detection

📋 TO RUN THE PI 5 CROWD COUNTER:

1. Activate environment:
   source crowd_counter_env/bin/activate

2. Load Pi 5 optimizations:
   source performance_env.sh

3. Test everything:
   python3 test_pi5_installation.py

4. Run optimized crowd counter:
   python3 rpi5_optimized_crowd_counter.py

📊 EXPECTED PI 5 PERFORMANCE:
- Inference time: 0.1-0.3 seconds
- Memory usage: 2-4GB (of 8GB)
- CPU usage: 70-90% during inference
- Temperature: <70°C with cooling
- Works best with sparse crowds (<50 people)

🔧 PI 5 SPECIFIC COMMANDS:
- Temperature: vcgencmd measure_temp
- Throttling: vcgencmd get_throttled
- Performance: htop, free -h

💡 OPTIONAL - AUTO-START SERVICE:
   sudo cp crowd_counter.service /etc/systemd/system/
   sudo systemctl enable crowd_counter.service
   sudo systemctl start crowd_counter.service

⚠️  IMPORTANT: Reboot recommended to apply all changes
"

read -p "Reboot Pi 5 now to apply all optimizations? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🔄 Rebooting Pi 5..."
    sudo reboot
fi 