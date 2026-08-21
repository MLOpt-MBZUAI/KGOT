"""
Data Preparation Script for KGOT.

Downloads and preprocesses:
1. PrimeKG (comprehensive biomedical KG, integrates multiple sources)
2. DUD-E benchmark (virtual screening evaluation)
3. Pre-computes molecule/protein embeddings

Reference: Section 3.1, Appendix B-D.
"""

import os
import sys
import urllib.request
from pathlib import Path
import pandas as pd
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.build_kg import KnowledgeGraph, build_kg_from_sources
from data.encoders import MoleculeEncoder, ProteinEncoder, precompute_embeddings


def download_primekg(data_dir: str):
    """
    Download PrimeKG — a comprehensive biomedical knowledge graph.
    Contains: drugs, proteins, genes, diseases, biological processes, pathways.
    ~130K nodes, ~4M edges, 30+ relation types.

    Source: Chandak et al., 2023, Nature Scientific Data.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    output_file = data_dir / 'primekg.csv'

    if output_file.exists():
        print(f"PrimeKG already exists: {output_file}")
        return output_file

    print("Downloading PrimeKG (comprehensive biomedical KG)...")
    print("  Source: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/IXA7BM")

    # PrimeKG direct download link
    url = "https://dataverse.harvard.edu/api/access/datafile/6180620"

    try:
        urllib.request.urlretrieve(url, str(output_file))
        print(f"  Downloaded to: {output_file}")

        # Verify
        df = pd.read_csv(output_file, nrows=5)
        print(f"  Columns: {list(df.columns)}")
        print(f"  Sample rows: {len(df)}")

    except Exception as e:
        print(f"  Download failed: {e}")
        print(f"  Please download manually from the Harvard Dataverse link above.")

    return output_file


def download_dude(data_dir: str):
    """
    Download DUD-E benchmark data.

    DUD-E contains 102 protein targets with actives and decoys.
    We download the processed version suitable for our evaluation.
    """
    data_dir = Path(data_dir) / 'dude'
    data_dir.mkdir(parents=True, exist_ok=True)

    info_file = data_dir / 'README.txt'

    if info_file.exists():
        print(f"DUD-E directory exists: {data_dir}")
        return

    print("DUD-E data preparation:")
    print("  The DUD-E dataset can be downloaded from:")
    print("  http://dude.docking.org/subsets/all")
    print()
    print("  After downloading, create a CSV file at:")
    print(f"  {data_dir / 'dude_data.csv'}")
    print("  With columns: smiles, target_id, label (1=active, 0=decoy)")
    print()
    print("  Alternatively, use the DrugCLIP processed version from:")
    print("  https://github.com/bowen-gao/DrugCLIP")

    # Create placeholder
    with open(info_file, 'w') as f:
        f.write("DUD-E Virtual Screening Benchmark\n")
        f.write("Download from: http://dude.docking.org/\n")
        f.write("Format: dude_data.csv with columns [smiles, target_id, label]\n")


def prepare_train_test_split(kg: KnowledgeGraph, n_test: int = 60000, seed: int = 42):
    """
    Create train/test split for link prediction.
    Withholds n_test molecule-protein pairs for evaluation.

    Reference: Section 3.2 — "we withhold 60,000 molecule-protein pairs"
    """
    np.random.seed(seed)

    # Find all molecule-protein triples
    mol_set = set(kg.molecule_ids)
    prot_set = set(kg.protein_ids)

    mpi_triples = []
    other_triples = []

    for h, r, t in kg.triples:
        if h in mol_set and t in prot_set:
            mpi_triples.append((h, r, t))
        elif t in mol_set and h in prot_set:
            mpi_triples.append((t, r, h))  # normalize to (mol, rel, prot)
        else:
            other_triples.append((h, r, t))

    print(f"  Total MPI triples: {len(mpi_triples)}")
    print(f"  Other triples: {len(other_triples)}")

    # Randomly select test pairs
    n_test = min(n_test, len(mpi_triples) // 5)  # max 20% for test
    indices = np.random.permutation(len(mpi_triples))
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]

    test_triples = [mpi_triples[i] for i in test_indices]
    train_mpi = [mpi_triples[i] for i in train_indices]

    # Training set = other_triples + train_mpi
    train_triples = other_triples + train_mpi

    print(f"  Train triples: {len(train_triples)} (MPI: {len(train_mpi)})")
    print(f"  Test triples: {len(test_triples)}")

    return (
        torch.tensor(train_triples, dtype=torch.long),
        torch.tensor(test_triples, dtype=torch.long),
        torch.tensor(train_mpi, dtype=torch.long),
    )


def main():
    """Run full data preparation pipeline."""
    base_dir = Path('./data')
    base_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("KGOT Data Preparation")
    print("=" * 60)

    # Step 1: Download PrimeKG
    print("\n--- Step 1: Download PrimeKG ---")
    download_primekg(str(base_dir / 'raw'))

    # Step 2: Build KG
    print("\n--- Step 2: Build Knowledge Graph ---")
    kg = build_kg_from_sources(str(base_dir / 'raw'), use_primekg=True)
    kg.save(str(base_dir / 'kg'))

    # Step 3: Train/Test split
    print("\n--- Step 3: Create Train/Test Split ---")
    train_triples, test_triples, train_mpi = prepare_train_test_split(kg)
    torch.save(train_triples, base_dir / 'train_triples.pt')
    torch.save(test_triples, base_dir / 'test_triples.pt')
    torch.save(train_mpi, base_dir / 'train_mpi_pairs.pt')

    # Step 4: DUD-E
    print("\n--- Step 4: Prepare DUD-E ---")
    download_dude(str(base_dir))

    # Step 5: Pre-compute embeddings (if Uni-Mol available)
    print("\n--- Step 5: Pre-compute Embeddings ---")
    # Collect molecule and protein IDs/names
    mol_names = [kg.id2entity[i] for i in kg.molecule_ids[:10000]]  # subset for speed
    prot_names = [kg.id2entity[i] for i in kg.protein_ids[:5000]]

    mol_encoder = MoleculeEncoder(use_unimol=False)  # fallback for now
    prot_encoder = ProteinEncoder(use_unimol=False)

    mol_embs, prot_embs = precompute_embeddings(
        mol_names, prot_names, mol_encoder, prot_encoder, str(base_dir / 'embeddings')
    )

    print("\n" + "=" * 60)
    print("Data preparation complete!")
    print(f"  KG: {base_dir / 'kg'}")
    print(f"  Triples: {base_dir / 'train_triples.pt'}")
    print(f"  Embeddings: {base_dir / 'embeddings'}")
    print("=" * 60)


if __name__ == '__main__':
    main()
