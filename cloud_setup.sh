#!/bin/bash
echo "=== Setting up Warehouse Robotics on Cloud ==="

# Install dependencies
pip install torch torchvision transformers numpy==1.26.4 \
    opencv-python gymnasium matplotlib tensorboard \
    tqdm pyyaml Pillow scipy stable-baselines3

# Install PyBullet
conda install -c conda-forge pybullet -y 2>/dev/null || pip install pybullet

# Create directories
mkdir -p data/demos_mobile checkpoints logs/cloud_bc logs/cloud_rl

# Verify
python3 -c "
import torch, pybullet, transformers
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
print('PyBullet: OK')
print('All dependencies ready!')
"

echo "=== Setup complete! ==="
echo "Run in order:"
echo "1. python3 src/data/collect_demos_cloud.py"
echo "2. python3 src/training/train_bc_cloud.py"
echo "3. python3 src/training/train_rl_cloud.py"
