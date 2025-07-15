# Raspberry Pi 5 Crowd Counting System

Real-time crowd counting using LWCC (Lightweight Crowd Counting) optimized for Raspberry Pi 5 with 8GB RAM.

## 🚀 Features

- **Real-time Processing**: Captures images every 10 seconds and performs crowd counting
- **Pi 5 Optimized**: Uses ARM-specific optimizations for maximum performance
- **Multiple Models**: Supports CSRNet, Bayesian, DM-Count, and SFANet models
- **Camera Integration**: Works with Pi Camera and USB cameras
- **Performance Monitoring**: Real-time CPU, memory, and inference time monitoring
- **Automatic Saving**: Saves results with density visualizations
- **Easy Setup**: Automated installation script

## 📋 Requirements

### Hardware
- **Raspberry Pi 5** (8GB RAM recommended, 4GB minimum)
- **Pi Camera Module** or USB camera
- **MicroSD Card** (32GB+, Class 10 or better)
- **Power Supply** (5V 3A official adapter)
- **Heatsink/Fan** (recommended for sustained performance)

### Software
- **64-bit Raspberry Pi OS** (Bookworm or later)
- **Python 3.9+**
- **Internet connection** for initial setup

## 🔧 Quick Start

### 1. Clone and Setup
```bash
git clone <your-repo-url>
cd RTSAComp

# Make setup script executable
chmod +x setup_rpi5.sh

# Run the setup script
./setup_rpi5.sh
```

### 2. Test Installation
```bash
# Activate virtual environment
source crowd_counter_env/bin/activate

# Load performance optimizations
source performance_env.sh

# Test the installation
python3 test_installation.py
```

### 3. Run Crowd Counting
```bash
# Basic usage (10-second intervals)
python3 rpi_crowd_counter.py

# Custom interval (30 seconds)
python3 rpi_crowd_counter.py --interval 30

# Different model (for dense crowds)
python3 rpi_crowd_counter.py --model CSRNet --weights SHA

# Run without saving results
python3 rpi_crowd_counter.py --no-save
```

## 📊 Performance Expectations

Based on Raspberry Pi 5 testing:

| Metric | Expected Value |
|--------|---------------|
| **Inference Time** | 0.1-0.3 seconds |
| **Memory Usage** | 2-4GB |
| **CPU Usage** | 70-90% during inference |
| **Best Performance** | Sparse crowds (<50 people) |
| **Recommended Model** | CSRNet with SHB weights |

## 🎯 Model Selection Guide

| Model | Weights | Best For | Performance |
|-------|---------|----------|-------------|
| CSRNet | SHB | Sparse crowds (<50) | ⭐⭐⭐⭐⭐ |
| CSRNet | SHA | Dense crowds (>50) | ⭐⭐⭐⭐ |
| DM-Count | SHB | Sparse crowds | ⭐⭐⭐⭐ |
| DM-Count | SHA | Dense crowds | ⭐⭐⭐ |
| Bay | SHB | Sparse crowds | ⭐⭐⭐ |
| SFANet | SHB | Sparse crowds | ⭐⭐⭐ |

## 🛠️ Advanced Configuration

### Environment Variables
```bash
# Performance optimizations (already set in performance_env.sh)
export DNNL_DEFAULT_FPMATH_MODE=BF16    # Enable BFloat16
export LRU_CACHE_CAPACITY=1024          # Cache optimization
export THP_MEM_ALLOC_ENABLE=1           # Transparent huge pages
export OMP_NUM_THREADS=4                # Optimize for Pi 5 cores
```

### Command Line Options
```bash
python3 rpi_crowd_counter.py --help

Options:
  --model {CSRNet,Bay,DM-Count,SFANet}  Model to use
  --weights {SHA,SHB,QNRF}              Model weights
  --interval INT                        Capture interval (seconds)
  --image-size INT INT                  Image size for inference
  --no-save                             Don't save results
  --gui                                 Display GUI (if available)
```

### System Service (Auto-start)
```bash
# Copy service file
sudo cp crowd_counter.service /etc/systemd/system/

# Enable auto-start
sudo systemctl enable crowd_counter.service

# Start service
sudo systemctl start crowd_counter.service

# Check status
sudo systemctl status crowd_counter.service

# View logs
journalctl -u crowd_counter -f
```

## 📁 Output Files

Results are saved in the `crowd_counting_results/` directory:

- **`crowd_count_YYYYMMDD_HHMMSS.png`**: Visualization with original image and density heatmap
- **`data_YYYYMMDD_HHMMSS.npz`**: Raw data (image, count, density map)

## 🔍 Monitoring and Troubleshooting

### Performance Monitoring
```bash
# Real-time system monitoring
htop

# Memory usage
free -h

# Temperature monitoring
vcgencmd measure_temp

# Service logs
journalctl -u crowd_counter -f
```

### Common Issues

#### 1. Camera Not Detected
```bash
# Check camera connection
ls /dev/video*

# Test camera
python3 -c "import cv2; cap = cv2.VideoCapture(0); print(cap.isOpened())"

# Enable camera in config
sudo raspi-config  # -> Interface Options -> Camera
```

#### 2. Memory Issues
```bash
# Check memory usage
free -h

# Increase swap if needed
sudo dphys-swapfile swapoff
sudo nano /etc/dphys-swapfile  # Set CONF_SWAPSIZE=2048
sudo dphys-swapfile setup
sudo dphys-swapfile swapon
```

#### 3. Performance Issues
```bash
# Check CPU governor
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor

# Set performance mode
echo 'performance' | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Check temperature throttling
vcgencmd get_throttled
```

## 🔧 Hardware Optimization Tips

### 1. Cooling
- Use heatsinks on CPU and memory
- Consider active cooling (fan) for sustained performance
- Monitor temperature: `vcgencmd measure_temp`

### 2. Power Supply
- Use official 5V 3A adapter
- Poor power can cause throttling
- Check for power issues: `vcgencmd get_throttled`

### 3. Storage
- Use fast microSD (Class 10, A1/A2 rated)
- Consider SSD via USB for better I/O
- Monitor disk usage: `df -h`

### 4. Memory
- Close unnecessary services
- Use `htop` to monitor memory usage
- Consider increasing swap size

## 🐛 Debugging

### Enable Debug Logging
```python
# Add to your script
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Performance Profiling
```bash
# Profile the application
python3 -m cProfile -o profile.stats rpi_crowd_counter.py

# View profiling results
python3 -c "import pstats; p = pstats.Stats('profile.stats'); p.sort_stats('cumulative'); p.print_stats(20)"
```

## 📚 Technical Details

### Model Architecture
- **CSRNet**: Convolutional Neural Network for crowd counting
- **SHB Weights**: Trained on Shanghai Part B dataset (sparse crowds)
- **SHA Weights**: Trained on Shanghai Part A dataset (dense crowds)
- **Input Size**: 224x224 RGB images
- **Output**: Density map and total count

### Optimizations Applied
- **QNNPACK**: ARM-optimized quantization engine
- **BFloat16**: Reduced precision for faster inference
- **Thread Optimization**: Configured for Pi 5's 4 cores
- **Memory Pooling**: Efficient memory allocation
- **Model Caching**: Pre-load models to reduce latency

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test on Raspberry Pi 5
5. Submit a pull request

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [LWCC Library](https://github.com/tersekmatija/lwcc) for the crowd counting models
- [PyTorch](https://pytorch.org/) for the deep learning framework
- [Raspberry Pi Foundation](https://www.raspberrypi.org/) for the hardware platform

## 📞 Support

For issues and questions:
1. Check the troubleshooting section above
2. Review the logs: `journalctl -u crowd_counter -f`
3. Test with: `python3 test_installation.py`
4. Open an issue in the repository

---

**Happy Crowd Counting! 🎉** 