#!/bin/bash

# Raspberry Pi 5 Crowd Counting Setup Script (Fixed Version)
# This script sets up the Pi 5 for optimal crowd counting performance

set -e

echo "🔧 Setting up Raspberry Pi 5 for Crowd Counting..."

# Check if we're on a Pi 5
if [[ $(cat /proc/device-tree/model) != *"Raspberry Pi 5"* ]]; then
    echo "❌ This script is designed for Raspberry Pi 5"
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

# Update system
echo "📦 Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install system dependencies (removed obsolete Qt4 packages)
echo "🔧 Installing system dependencies..."
sudo apt install -y \
    python3-pip \
    python3-venv \
    python3-dev \
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
    libssl-dev

# Enable camera
echo "📷 Configuring camera..."
sudo raspi-config nonint do_camera 0

# Create virtual environment
echo "🐍 Creating Python virtual environment..."
python3 -m venv crowd_counter_env
source crowd_counter_env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch with CPU-only support (optimized for ARM)
echo "🔥 Installing PyTorch (CPU-only, ARM optimized)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other Python dependencies
echo "📚 Installing Python dependencies..."
pip install -r requirements_rpi.txt

# Performance optimizations
echo "⚡ Applying performance optimizations..."

# Enable performance governor
echo 'performance' | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Configure memory split for camera
if ! grep -q "gpu_mem=128" /boot/firmware/config.txt; then
    echo "gpu_mem=128" | sudo tee -a /boot/firmware/config.txt
fi

# Enable camera
if ! grep -q "camera_auto_detect=1" /boot/firmware/config.txt; then
    echo "camera_auto_detect=1" | sudo tee -a /boot/firmware/config.txt
fi

# Set up swap for additional memory during model loading
echo "💾 Setting up swap file..."
if [ ! -f /swapfile ]; then
    sudo fallocate -l 4G /swapfile
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile
    sudo swapon /swapfile
    echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
fi

# Create performance environment script
cat > performance_env.sh << 'EOF'
#!/bin/bash
# Performance environment variables for Raspberry Pi 5 crowd counting

# PyTorch optimizations
export DNNL_DEFAULT_FPMATH_MODE=BF16
export LRU_CACHE_CAPACITY=1024
export THP_MEM_ALLOC_ENABLE=1
export OMP_NUM_THREADS=4

# ARM-specific optimizations
export PYTORCH_THREAD_LOCAL_MEMORY_POOL=1
export PYTORCH_QNNPACK_RUNTIME_QUANTIZATION=1

# Memory optimizations
export MALLOC_TRIM_THRESHOLD_=100000
export MALLOC_MMAP_THRESHOLD_=100000

echo "Performance environment configured for Pi 5"
EOF

chmod +x performance_env.sh

# Create systemd service for automatic startup (optional)
cat > crowd_counter.service << 'EOF'
[Unit]
Description=Raspberry Pi 5 Crowd Counter
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/RTSA_competition
Environment=PATH=/home/pi/RTSA_competition/crowd_counter_env/bin
ExecStartPre=/home/pi/RTSA_competition/performance_env.sh
ExecStart=/home/pi/RTSA_competition/crowd_counter_env/bin/python /home/pi/RTSA_competition/rpi_crowd_counter.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Make the main script executable
chmod +x rpi_crowd_counter.py

echo "🎯 Testing PyTorch installation..."
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CPU cores: {torch.get_num_threads()}')"

echo "🧪 Testing LWCC installation..."
python3 -c "
try:
    from lwcc import LWCC
    print('✅ LWCC imported successfully')
except ImportError as e:
    print(f'⚠️  LWCC import failed: {e}')
    print('Installing LWCC manually...')
    import subprocess
    subprocess.run(['pip', 'install', 'lwcc', '--no-cache-dir'])
    try:
        from lwcc import LWCC
        print('✅ LWCC installed and imported successfully')
    except ImportError as e2:
        print(f'❌ LWCC installation failed: {e2}')
"

echo "📋 Testing camera access..."
python3 -c "
try:
    import cv2
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        print('✅ Camera access successful')
        cap.release()
    else:
        print('⚠️  Camera access failed - check camera connection')
except Exception as e:
    print(f'⚠️  Camera test failed: {e}')
"

echo "
🚀 Setup complete! 

To run the crowd counter:
1. Activate the virtual environment:
   source crowd_counter_env/bin/activate
   
2. Load performance optimizations:
   source performance_env.sh
   
3. Run the crowd counter:
   python3 rpi_crowd_counter.py
   
Optional arguments:
  --interval 10           # Capture every 10 seconds
  --model CSRNet         # Use CSRNet model
  --weights SHB          # Use SHB weights (good for sparse crowds)
  --image-size 224 224   # Input image size
  --no-save              # Don't save results
  --gui                  # Display GUI (if available)

For automatic startup:
  sudo cp crowd_counter.service /etc/systemd/system/
  sudo systemctl enable crowd_counter.service
  sudo systemctl start crowd_counter.service

Performance monitoring:
  htop                   # Monitor CPU/memory usage
  journalctl -u crowd_counter -f  # View logs

🔧 Hardware recommendations:
- Use heatsinks/fan for sustained performance
- Ensure adequate power supply (5V 3A)
- Use fast microSD card (Class 10 or better)
- Consider SSD for better I/O performance

📊 Expected Performance:
- Inference time: ~0.1-0.3 seconds per image
- Memory usage: ~2-4GB
- CPU usage: ~70-90% during inference
- Works best with sparse crowds (<50 people)

Reboot recommended to apply all changes.
"

read -p "Reboot now? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    sudo reboot
fi 