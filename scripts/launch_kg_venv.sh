#!/bin/bash
# Launcher for the ankith box (venv, not conda). Env overrides:
# KG_GPU, KG_DIM, KG_EPOCHS, KG_NNEG, KG_GAMMA, KG_MODELS, KG_TAG.
set -e
cd ~/KGOT
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=${KG_GPU:-0}
export KG_DIM=${KG_DIM:-128}
export KG_EPOCHS=${KG_EPOCHS:-50}
export KG_NNEG=${KG_NNEG:-32}
export KG_GAMMA=${KG_GAMMA:-9.0}
export KG_MODELS=${KG_MODELS:-rotate}
export KG_TAG=${KG_TAG:-rotate}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "Launch $(date) GPU=${CUDA_VISIBLE_DEVICES} DIM=${KG_DIM} EP=${KG_EPOCHS} NNEG=${KG_NNEG} GAMMA=${KG_GAMMA} MODELS=${KG_MODELS}"
python -u scripts/run_kg_training.py
echo "Done $(date)"
