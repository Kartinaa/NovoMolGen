#!/bin/bash
#SBATCH --partition=main
#SBATCH -J install-requirements
#SBATCH --output=%x.%j.out
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=12:00:00
#SBATCH --exclude=cn-k[001-002]
set -e
set -v

FLASH_ATTN_VERSION='2.6.1'
export MAX_JOBS=4

# Default config
if [ -z "${WORK_DIR}" ]; then
    WORK_DIR=$SLURM_TMPDIR/workspace
fi
mkdir -p $WORK_DIR


module load anaconda/3
module load cuda/11.8

conda create -c conda-forge -c r -c msys2 -c lich -c hcc -n NovoMol python=3.10 openbabel openmm pdbfixer rdkit syba xtb xtb-python crest lightgbm=4.3.0 deepsmiles=1.0.1
conda activate NovoMol
pip install --upgrade pip

# Note: Ensure that the installed PyTorch version supports CUDA 11.8
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118

# Note: Ensure that ninja is uninstalled and reinstalled to avoid conflicts
pip uninstall -y ninja && pip install ninja

# Install dependencies from requirements.txt
pip install -r requirements.txt

# Clone and install flash-attention v2
NV_CC="8.0;8.6" # flash-attention-v2 and exllama_kernels are anyway limited to CC of 8.0+
FLASH_ATTENTION_DIR="$WORK_DIR/flash-attention-v2"
git clone https://github.com/Dao-AILab/flash-attention "$FLASH_ATTENTION_DIR"
pushd "$FLASH_ATTENTION_DIR"
git checkout "tags/v$FLASH_ATTN_VERSION"
TORCH_CUDA_ARCH_LIST="$NV_CC" MAX_JOBS="$MAX_JOBS" python setup.py install
pushd csrc/fused_dense_lib && pip install .
pushd ../xentropy && pip install .
pushd ../rotary && pip install .
pushd ../layer_norm && pip install .
popd  # Exit from csrc/rotary
popd  # Exit from flash-attention
