#!/bin/bash
#SBATCH --job-name=dinov3_extract
#SBATCH --partition=braid
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --output=/braid/vanasseh/dinov3_inet22k_embeddings/extraction_%j.log
#SBATCH --error=/braid/vanasseh/dinov3_inet22k_embeddings/extraction_%j.err

# Log job information
echo "Starting DINOv3 embedding extraction job"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Date: $(date)"
echo "Working directory: $(pwd)"

# Activate conda environment
source /homefs/home/vanasseh/miniforge3/bin/activate
conda activate torchdr_venv

# Show environment info
echo "Python location: $(which python)"
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "CUDA device: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")')"

# Set environment variables for better performance
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Change to extraction directory
cd /braid/vanasseh/dinov3_inet22k_embeddings/

# Run extraction
echo "Starting extraction at $(date)"
python extract_single_gpu.py \
    --model "facebook/dinov3-vit7b16-pretrain-lvd1689m" \
    --batch-size 64 \
    --num-workers 4 \
    --output "/braid/vanasseh/dinov3_inet22k_embeddings/embeddings.h5"

# Check exit status
if [ $? -eq 0 ]; then
    echo "Extraction completed successfully at $(date)"

    # Print file info
    echo "Output file information:"
    ls -lh /braid/vanasseh/dinov3_inet22k_embeddings/embeddings.h5

    # Quick validation with Python
    python -c "
import h5py
import sys
try:
    with h5py.File('/braid/vanasseh/dinov3_inet22k_embeddings/embeddings.h5', 'r') as f:
        print(f'Total images: {f[\"metadata\"].attrs.get(\"total_images\", \"N/A\")}')
        print(f'Embedding dim: {f[\"metadata\"].attrs.get(\"embedding_dim\", \"N/A\")}')
        print(f'Embeddings shape: {f[\"embeddings\"].shape}')
        print(f'Keys shape: {f[\"image_keys\"].shape}')
except Exception as e:
    print(f'Error reading HDF5 file: {e}')
    sys.exit(1)
"
else
    echo "Extraction failed with exit code $?"
    exit 1
fi

echo "Job completed at $(date)"
