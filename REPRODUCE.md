# Reproducing KGOT

End-to-end pipeline for **KGOT: Unified Knowledge Graph & Optimal Transport
Pseudo-Labeling for Molecule–Protein Interaction Prediction**. Runs on a fresh
Linux box with an NVIDIA GPU. Total wall time is dominated by data download
(~8 GB) and KG training.

## 0. Environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# GPU: if torch didn't pick up CUDA, install the CUDA build matching your driver
#   pip install torch --index-url https://download.pytorch.org/whl/cu121
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```
Tested with Python 3.10–3.12, PyTorch 2.x (CUDA), on A100 GPUs.
Disk: keep ~30 GB free (raw downloads + intermediate `.npy`/checkpoints).

## 1. One-shot run

```bash
bash scripts/download_all.sh        # fetch all 8 raw sources into data/raw/
bash scripts/run_all.sh             # build KG -> embeddings -> score -> link prediction
```
`run_all.sh` runs the 5 KG models with a single margin (γ=9). **TorusE needs γ≈4**
(its torus score range is small); run it separately for a good result:
```bash
KG_MODELS=toruse KG_TAG=toruse KG_GAMMA=4.0 KG_DIM=256 python -u scripts/run_kg_training.py
```

## 2. What each stage does (and its output)

| Stage | Script | Output |
|---|---|---|
| Download | `scripts/download_all.sh` | `data/raw/{kegg,enzyme,go,primekg,pfam,idmap,pubchem}` |
| Build KG (reconciled) | `scripts/build_reconciled_kg.py` | `data/kg/kg_data.pt` (KEGG+ENZYME+GO+PrimeKG; NCBI→UniProt) |
| Merge PFam | `scripts/merge_pfam.py` | appends ~9.1M protein–family edges (→ ~10.5M triples) |
| To numpy | `scripts/preprocess_kg.py` | `triples.npy`, `molecule_ids.npy`, `protein_ids.npy`, `metadata.pt` |
| MPI id lists | `scripts/prep_mpi_ids.py` | `mpi_molecule_ids.npy`, `mpi_protein_ids.npy`, `mpi_{drug,prot}_map.pt` |
| Molecule emb | `scripts/build_mol_ftp.py` | `data/embeddings/mol_embeddings.pt` (Morgan→512), `mol_kg_ids.npy` |
| Protein emb | `scripts/build_prot_embeddings.py` | `data/embeddings/prot_embeddings.pt` (ESM-2→512), `prot_kg_ids.npy` |
| Re-key emb | `scripts/rekey_embeddings.py` | remaps `*_kg_ids.npy` to reconciled entity ids |
| Score fn | `scripts/train_score_infonce.py` | `data/embeddings/score_model.pt` |
| Link prediction | `scripts/run_kg_training.py` | `data/kg/link_pred_<tag>.pt`, Hits@K on stdout |

## 3. Method → code map

- **Score function** `S(x,y)=W(f(x)⊕g(y))` — `models/score_model.py`
- **Inverse-OT / InfoNCE training** — `scripts/train_score_infonce.py`
- **Sinkhorn OT + pseudo-label extraction** — `utils/sinkhorn.py` (used in `run_kg_training.py`)
- **KG embedding models** (PairRE, RotatE, MuRE, TorusE, ComplEx-FF) — `models/kg_embeddings.py`
- **KG augmentation + `L_total = L_KG + α·L_pseudo`** — `scripts/run_kg_training.py`
- **KG construction** — `data/build_kg.py`, `scripts/build_reconciled_kg.py`, `scripts/merge_pfam.py`

## 4. Evaluation protocol (link prediction)

`run_kg_training.py` uses the aligned protocol: for each held-out MPI edge, rank the
true protein against a candidate pool (`KG_CANDMAX`, default 50k) sampled from all
protein entities, **filtered** (other known partners masked). Training negatives are
sampled from the full protein set (`KG_NEGPOOL=full`) with self-adversarial weighting.

Config knobs (env vars): `KG_DIM`, `KG_EPOCHS`, `KG_NNEG`, `KG_GAMMA`, `KG_MODELS`,
`KG_TAG`, `KG_CANDMAX`, `KG_NEGPOOL`. GPU via `CUDA_VISIBLE_DEVICES`.

## 5. Notes / gotchas

- **PubChem REST is rate-limited**; molecule SMILES are fetched from PubChem **FTP**
  bulk files (`build_mol_ftp.py`) which are not throttled.
- **Memory:** the entity table scales with #entities (~9M with PFam). On 40 GB GPUs,
  RotatE/ComplEx (double-size tables) need `KG_DIM<=128`; single-table models
  (TorusE/PairRE/MuRE) fit up to ~256. On 80 GB GPUs all fit at 256.
- Data sources evolve; `download_all.sh` pins working URLs (Pfam 37.4, PrimeKG
  Dataverse file 6180620). Update these if upstream moves them.
- Paper PDF: `paper/`. Slides & poster: `presentation/`.
