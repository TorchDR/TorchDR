#!/bin/bash
#SBATCH --job-name=download_tahoe_scvi
#SBATCH --output=/braid/vanasseh/tahoe_scvi_download_%j.log
#SBATCH --error=/braid/vanasseh/tahoe_scvi_download_%j.err
#SBATCH --partition=braid
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --ntasks=1

# Activate conda environment
source /homefs/home/vanasseh/miniforge3/bin/activate && conda activate torchdr_venv

# Create target directory
TARGET_DIR="/braid/vanasseh/tahoe-100m-scvi-v1"
mkdir -p ${TARGET_DIR}

# Change to target directory
cd ${TARGET_DIR}

echo "Starting download of Tahoe-100M-SCVI-v1 model and minified data at $(date)"
echo "Target directory: ${TARGET_DIR}"
echo "Python path: $(which python)"
echo "Conda env: $CONDA_DEFAULT_ENV"

# Install/upgrade huggingface-hub and hf-transfer if needed
pip install --upgrade huggingface-hub hf-transfer

# Download the model repository (contains both model and minified AnnData)
# Using resume download in case of interruption
export HF_HUB_ENABLE_HF_TRANSFER=1

# Download as a model repository (NOT dataset)
huggingface-cli download tahoebio/Tahoe-100M-SCVI-v1 \
    --local-dir . \
    --resume-download

echo "Download completed at $(date)"
echo "Checking downloaded files..."
ls -la ${TARGET_DIR}
du -sh ${TARGET_DIR}
