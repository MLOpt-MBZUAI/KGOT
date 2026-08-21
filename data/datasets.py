"""
Evaluation Datasets for KGOT.

1. DUD-E: Virtual screening benchmark (102 protein targets, 22886 actives)
2. LIT-PCBA: More challenging VS benchmark (7844 actives, 407381 inactives)
3. Link Prediction: 60K held-out molecule-protein pairs from KG

Reference: Section 3.1-3.2 of the paper.
"""

import os
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from torch.utils.data import Dataset


class DUDEDataset(Dataset):
    """
    DUD-E (Directory of Useful Decoys, Enhanced) dataset.

    102 protein targets, each with actives + decoys.
    Evaluation: rank actives above decoys for each target.

    Expected directory structure:
        dude/
        ├── targets.csv          # target_id, target_name, uniprot_id
        ├── actives/
        │   ├── target1.smi      # SMILES of active molecules
        │   └── ...
        └── decoys/
            ├── target1.smi      # SMILES of decoy molecules
            └── ...

    Or single file format:
        dude/dude_data.csv       # smiles, target_id, label (1=active, 0=decoy)
    """

    def __init__(self, data_dir: str, mol_encoder=None, prot_encoder=None):
        self.data_dir = Path(data_dir)
        self.mol_encoder = mol_encoder
        self.prot_encoder = prot_encoder

        self.targets = []  # list of target info dicts
        self.data = []     # list of (smiles, target_id, label) tuples

        self._load_data()

    def _load_data(self):
        """Load DUD-E data."""
        # Try single CSV format first
        csv_path = self.data_dir / 'dude_data.csv'
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                self.data.append({
                    'smiles': row['smiles'],
                    'target_id': row['target_id'],
                    'label': int(row['label']),
                })
            self.targets = list(df['target_id'].unique())
            print(f"  DUD-E loaded: {len(self.data)} pairs, {len(self.targets)} targets")
            return

        # Try directory structure
        targets_file = self.data_dir / 'targets.csv'
        if targets_file.exists():
            targets_df = pd.read_csv(targets_file)
            for _, row in targets_df.iterrows():
                target_id = row['target_id']
                self.targets.append(target_id)

                # Load actives
                actives_file = self.data_dir / 'actives' / f'{target_id}.smi'
                if actives_file.exists():
                    with open(actives_file) as f:
                        for line in f:
                            smi = line.strip().split()[0]
                            self.data.append({'smiles': smi, 'target_id': target_id, 'label': 1})

                # Load decoys
                decoys_file = self.data_dir / 'decoys' / f'{target_id}.smi'
                if decoys_file.exists():
                    with open(decoys_file) as f:
                        for line in f:
                            smi = line.strip().split()[0]
                            self.data.append({'smiles': smi, 'target_id': target_id, 'label': 0})

        if not self.data:
            print(f"  DUD-E: No data found at {self.data_dir}")
            print(f"  Download from: http://dude.docking.org/")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def get_target_data(self, target_id) -> Tuple[List[str], List[int]]:
        """Get all molecules and labels for a specific target."""
        smiles = []
        labels = []
        for item in self.data:
            if item['target_id'] == target_id:
                smiles.append(item['smiles'])
                labels.append(item['label'])
        return smiles, labels


class LITPCBADataset(Dataset):
    """
    LIT-PCBA dataset.

    More challenging than DUD-E: 7844 actives, 407381 inactives.
    Extreme class imbalance.

    Format: CSV with columns [smiles, target_id, label]
    Download: https://drugdesign.unistra.fr/LIT-PCBA/
    """

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.data = []
        self.targets = []

        csv_path = self.data_dir / 'litpcba_data.csv'
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                self.data.append({
                    'smiles': row['smiles'],
                    'target_id': row['target_id'],
                    'label': int(row['label']),
                })
            self.targets = list(df['target_id'].unique())
            print(f"  LIT-PCBA loaded: {len(self.data)} pairs, {len(self.targets)} targets")
        else:
            print(f"  LIT-PCBA: No data at {csv_path}")
            print(f"  Download from: https://drugdesign.unistra.fr/LIT-PCBA/")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class LinkPredictionDataset(Dataset):
    """
    Link prediction evaluation dataset.

    60K held-out molecule-protein pairs from the KG.
    Task: given a molecule, rank candidate proteins (head entity prediction).

    Metrics: Hits@1, Hits@3, Hits@5
    """

    def __init__(self, test_pairs: torch.Tensor, kg_triples: torch.Tensor,
                 num_entities: int, num_proteins: int):
        """
        Args:
            test_pairs: (N_test, 3) tensor of (mol_id, relation_id, prot_id)
            kg_triples: all training triples (for filtering)
            num_entities: total number of entities in KG
            num_proteins: number of protein entities
        """
        self.test_pairs = test_pairs
        self.kg_triples = kg_triples
        self.num_entities = num_entities
        self.num_proteins = num_proteins

        # Build filter set (true triples to exclude from negative ranking)
        self.filter_set = set()
        for triple in kg_triples.tolist():
            self.filter_set.add(tuple(triple))

    def __len__(self):
        return len(self.test_pairs)

    def __getitem__(self, idx):
        return self.test_pairs[idx]

    def evaluate(self, model, protein_ids: List[int], device: torch.device, batch_size: int = 256) -> Dict[str, float]:
        """
        Evaluate link prediction: for each test triple (h, r, t),
        rank all candidate proteins and find rank of true tail.
        """
        from utils.metrics import evaluate_link_prediction

        model.eval()
        ranks = []

        protein_tensor = torch.tensor(protein_ids, dtype=torch.long, device=device)

        with torch.no_grad():
            for i in range(0, len(self.test_pairs), batch_size):
                batch = self.test_pairs[i:i+batch_size].to(device)

                for triple in batch:
                    h, r, t = triple[0], triple[1], triple[2]

                    # Score all candidate proteins
                    heads = h.expand(len(protein_ids))
                    rels = r.expand(len(protein_ids))

                    scores = model(heads, rels, protein_tensor)

                    # Get rank of true tail
                    true_score = model(h.unsqueeze(0), r.unsqueeze(0), t.unsqueeze(0))
                    rank = (scores >= true_score).sum().item()
                    ranks.append(rank)

        return evaluate_link_prediction(ranks)


class MPITrainDataset(Dataset):
    """
    Training dataset for the score function.
    Contains labeled molecule-protein pairs (positive and negative).
    """

    def __init__(self, mol_embeddings: torch.Tensor, prot_embeddings: torch.Tensor,
                 positive_pairs: torch.Tensor):
        """
        Args:
            mol_embeddings: (M, d) pre-computed molecule embeddings
            prot_embeddings: (N, d) pre-computed protein embeddings
            positive_pairs: (K, 2) tensor of (mol_idx, prot_idx) positive pairs
        """
        self.mol_embeddings = mol_embeddings
        self.prot_embeddings = prot_embeddings
        self.positive_pairs = positive_pairs
        self.num_molecules = mol_embeddings.size(0)
        self.num_proteins = prot_embeddings.size(0)

    def __len__(self):
        return len(self.positive_pairs)

    def __getitem__(self, idx):
        mol_idx, prot_idx = self.positive_pairs[idx]
        return self.mol_embeddings[mol_idx], self.prot_embeddings[prot_idx]
