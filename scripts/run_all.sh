#!/bin/bash
# End-to-end KGOT reproduction. Run from the project root after download_all.sh.
#   bash scripts/run_all.sh
# Env knobs (see scripts/launch_kg_venv.sh): KG_GPU, KG_DIM, KG_EPOCHS, KG_NNEG, KG_GAMMA.
set -e
PY=${PYTHON:-python}

echo "== 1/7  Build KG (KEGG+ENZYME+GO+PrimeKG, NCBI->UniProt reconciled) =="
$PY -u scripts/build_reconciled_kg.py

echo "== 2/7  Merge PFam protein-family edges (capped ~9.1M) =="
$PY -u scripts/merge_pfam.py

echo "== 3/7  Convert KG to numpy (triples/ids/metadata) =="
$PY -u scripts/preprocess_kg.py

echo "== 4/7  Extract MPI molecule/protein id lists + name maps =="
$PY -u scripts/prep_mpi_ids.py

echo "== 5/7  Molecule embeddings (DrugBank->CID->SMILES via FTP -> Morgan 512) =="
$PY -u scripts/build_mol_ftp.py

echo "== 6/7  Protein embeddings (UniProt sequences -> ESM-2 650M -> 512) =="
$PY -u scripts/build_prot_embeddings.py
echo "        Re-key embeddings to reconciled entity ids"
$PY -u scripts/rekey_embeddings.py

echo "== 7/7  Train score function (InfoNCE) then KG link prediction =="
$PY -u scripts/train_score_infonce.py
KG_MODELS=${KG_MODELS:-rotate,pairre,mure,toruse,complex_ff} \
KG_DIM=${KG_DIM:-256} KG_EPOCHS=${KG_EPOCHS:-50} KG_NNEG=${KG_NNEG:-32} \
KG_TAG=${KG_TAG:-all} $PY -u scripts/run_kg_training.py

echo "Done. Link-prediction results in data/kg/link_pred_*.pt and stdout."
