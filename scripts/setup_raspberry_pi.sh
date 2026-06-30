#!/usr/bin/env bash
# One-time OS-level setup for Raspberry Pi 5 (Raspberry Pi OS 64-bit / Bookworm+).
# Run as the deployment user (not root) with sudo available: bash scripts/setup_raspberry_pi.sh
set -euo pipefail

echo "== Updating system packages =="
sudo apt-get update
sudo apt-get upgrade -y

echo "== Installing system dependencies =="
sudo apt-get install -y \
    python3-venv python3-pip python3-dev \
    libopenblas-dev libatlas-base-dev \
    libjpeg-dev libpng-dev \
    libcamera-apps libcamera-dev python3-libcamera python3-picamera2 \
    mosquitto mosquitto-clients \
    git sqlite3

echo "== Enabling camera interface =="
sudo raspi-config nonint do_camera 0 || echo "raspi-config camera toggle skipped (already enabled or headless config)"

echo "== Setting GPU memory split for camera ISP =="
CONFIG_FILE="/boot/firmware/config.txt"
if [ -f "$CONFIG_FILE" ] && ! grep -q "^gpu_mem=" "$CONFIG_FILE"; then
    echo "gpu_mem=128" | sudo tee -a "$CONFIG_FILE"
fi

echo "== Configuring swap (8GB RAM is enough at idle, but keep 1GB swap as a safety margin for"
echo "   the rare burst - e.g. ONNX Runtime graph-load - rather than letting the OOM killer fire) =="
if [ -f /etc/dphys-swapfile ]; then
    sudo dphys-swapfile swapoff || true
    sudo sed -i 's/^CONF_SWAPSIZE=.*/CONF_SWAPSIZE=1024/' /etc/dphys-swapfile
    sudo dphys-swapfile setup
    sudo dphys-swapfile swapon
fi

echo "== Setting CPU governor to 'ondemand' (balance power budget vs latency) =="
echo 'ondemand' | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor > /dev/null || true

echo "Setup complete. Next: bash scripts/install_dependencies.sh"
