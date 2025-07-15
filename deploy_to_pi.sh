#!/bin/bash

# Deployment script for Raspberry Pi 5
# Run this after git pull to ensure proper setup

set -e

echo "🚀 Deploying crowd counting system to Raspberry Pi 5..."

# Check if we're on Pi 5
if [[ $(cat /proc/device-tree/model) == *"Raspberry Pi 5"* ]]; then
    echo "✅ Running on Raspberry Pi 5"
else
    echo "⚠️  Not on Raspberry Pi 5, but continuing..."
fi

# Check for virtual environment
if [ ! -d "crowd_counter_env" ]; then
    echo "🔧 Virtual environment not found, running full setup..."
    ./setup_rpi5.sh
    exit 0
fi

# Activate virtual environment
source crowd_counter_env/bin/activate

# Update Python dependencies
echo "📦 Updating Python dependencies..."
pip install --upgrade -r requirements_rpi.txt

# Make scripts executable
chmod +x *.sh *.py

# Load performance environment
source performance_env.sh

# Test the installation
echo "🧪 Testing installation..."
python3 test_installation.py

# Restart service if it's running
if systemctl is-active --quiet crowd_counter; then
    echo "🔄 Restarting crowd counter service..."
    sudo systemctl restart crowd_counter
    echo "✅ Service restarted"
else
    echo "ℹ️  Service not running, you can start it manually"
fi

echo "🎉 Deployment complete!"
echo ""
echo "To run manually:"
echo "  source crowd_counter_env/bin/activate"
echo "  source performance_env.sh"
echo "  python3 rpi_crowd_counter.py"
echo ""
echo "To start as service:"
echo "  sudo systemctl start crowd_counter"
echo ""
echo "To check status:"
echo "  sudo systemctl status crowd_counter" 