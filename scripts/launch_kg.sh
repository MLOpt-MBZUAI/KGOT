#!/bin/bash
# Launch KG training. Env overrides: KG_GPU, KG_DIM, KG_EPOCHS, KG_NNEG, KG_NEVAL, KG_MODELS, KG_TAG, KG_LOG.
set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /efs/jiayuqin/conda_envs/kvcomm
cd ~/KGOT
export CUDA_VISIBLE_DEVICES=${KG_GPU:-4}
export KG_DIM=${KG_DIM:-256}
export KG_EPOCHS=${KG_EPOCHS:-80}
export KG_NNEG=${KG_NNEG:-64}
export KG_GAMMA=${KG_GAMMA:-9.0}
export KG_NEVAL=${KG_NEVAL:-2000}
export KG_MODELS=${KG_MODELS:-rotate,pairre,mure,toruse,complex_ff}
export KG_TAG=${KG_TAG:-all}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "Launch $(date) GPU=${CUDA_VISIBLE_DEVICES} DIM=${KG_DIM} EP=${KG_EPOCHS} NNEG=${KG_NNEG} GAMMA=${KG_GAMMA} MODELS=${KG_MODELS}"
python -u scripts/run_kg_training.py
echo "Done $(date)"
