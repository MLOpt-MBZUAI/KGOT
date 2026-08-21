# KGOT

Official release for **KGOT: Unified Knowledge Graph and Optimal Transport Pseudo-Labeling for Molecule-Protein Interaction Prediction** (ICLR 2026).

**Authors:** Jiayu Qin, Zhengquan Luo, Guy Tadmor, Changyou Chen, David Zeevi, and Zhiqiang Xu.

The paper is included as [`paper.pdf`](paper.pdf). A poster and talk deck are available under [`presentation/`](presentation/).

KGOT integrates heterogeneous biological resources into a unified knowledge graph, trains a molecule-protein score function, obtains pseudo-labels through optimal transport with similarity constraints, and augments knowledge graph embedding training with those pseudo-interactions.

## Method-to-code map

- Score function and inverse-OT/InfoNCE training: `models/score_model.py`, `scripts/train_score_infonce.py`
- Sinkhorn optimal transport: `utils/sinkhorn.py`
- PairRE, RotatE, MuRE, TorusE, and ComplEx-FF: `models/kg_embeddings.py`
- KG augmentation and link-prediction training: `scripts/run_kg_training.py`
- Reconciled KG construction: `scripts/build_reconciled_kg.py`, `scripts/merge_pfam.py`

## Quick start

Python 3.10–3.12 and a CUDA-capable PyTorch installation are recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
bash scripts/download_all.sh
bash scripts/run_all.sh
```

The pipeline downloads its source datasets, builds the reconciled KG, creates molecule and protein embeddings, trains the score model, and evaluates KG link prediction. Plan for approximately 30 GB of free disk space. Raw datasets, intermediate arrays, and checkpoints are not committed.

See [`REPRODUCE.md`](REPRODUCE.md) for exact stage outputs, model-specific settings, memory guidance, and evaluation details.

## Data sources and evaluation

The pipeline uses public KEGG, ENZYME, Gene Ontology, PrimeKG, Pfam, UniProt mapping, and PubChem resources. It evaluates molecule-protein interaction prediction with filtered link-prediction metrics and supports virtual-screening data preparation for DUD-E and LIT-PCBA.

## Citation

Please cite the ICLR 2026 paper. The final BibTeX entry can be added once the OpenReview proceedings metadata is available.

