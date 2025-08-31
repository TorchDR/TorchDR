#!/bin/bash
# SLURM job script for downloading ImageNet-22k

# Request resources and start download in tmux
echo "Requesting SLURM resources..."
srun --partition=braid \
     --time=48:00:00 \
     --cpus-per-task=16 \
     --mem=50G \
     --job-name=imagenet22k-download \
     --pty tmux new-session -s imagenet22k-download \
     "source /homefs/home/vanasseh/miniforge3/bin/activate && \
      conda activate torchdr_venv && \
      python /homefs/home/vanasseh/TorchDR/download_imagenet22k.py --max-workers 16"
