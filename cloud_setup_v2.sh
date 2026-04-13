#!/bin/bash
# Cloud setup for Phase 2 RL training on Vast.ai
# Run this on a fresh Vast.ai instance (RTX 3090/4090 recommended)

set -e

echo "=== Phase 2 Cloud Training Setup ==="

# Clone repo
if [ ! -d "warehouse_robotics" ]; then
    git clone https://github.com/Rajiv308/warehouse_robotics.git
fi
cd warehouse_robotics

# Install Python dependencies
pip install --quiet pybullet torch torchvision transformers numpy pyyaml tqdm tensorboard opencv-python-headless

# Test EGL GPU rendering
echo ""
echo "=== Testing EGL GPU Rendering ==="
python3 -c "
import pybullet as p
cid = p.connect(p.DIRECT)
try:
    egl = p.loadPlugin('eglRendererPlugin')
    if egl >= 0:
        print('EGL GPU rendering: ENABLED (10-50x faster)')
    else:
        print('WARNING: EGL plugin returned negative ID')
except Exception as e:
    print(f'WARNING: EGL not available: {e}')
    print('Falling back to CPU rendering (slower)')
p.disconnect()
"

# Test CUDA
echo ""
echo "=== Testing CUDA ==="
python3 -c "
import torch
if torch.cuda.is_available():
    print(f'CUDA: ENABLED - {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
else:
    print('WARNING: No CUDA GPU detected!')
"

# Check that BC checkpoint exists (needed to initialize RL)
echo ""
echo "=== Checking Checkpoints ==="
if [ -f "checkpoints/best_mobile_model.pth" ]; then
    echo "BC checkpoint found: checkpoints/best_mobile_model.pth"
    ls -lh checkpoints/best_mobile_model.pth
else
    echo "WARNING: No BC checkpoint! You need to copy best_mobile_model.pth to checkpoints/"
    echo "You can SCP it from your local machine:"
    echo "  scp checkpoints/best_mobile_model.pth <cloud_instance>:warehouse_robotics/checkpoints/"
fi

echo ""
echo "=== Setup Complete ==="
echo ""
echo "To start training:"
echo "  python3 src/training/train_rl_cloud.py"
echo ""
echo "To monitor (in another terminal):"
echo "  tensorboard --logdir logs/cloud_rl --bind_all"
