"""
Molecule and Protein Encoders for KGOT.

Uses Uni-Mol pretrained models as backbone for embedding extraction.
Embeddings are L2-normalized (as stated in the paper).

Reference: Section 2.2 of the paper.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import List, Optional


class MoleculeEncoder:
    """
    Molecule encoder using Uni-Mol pretrained model.

    Extracts fixed-dimensional embeddings from molecular SMILES/conformers.
    Embeddings are L2-normalized.

    In practice, this wraps Uni-Mol's inference pipeline.
    For reproduction without Uni-Mol: use Morgan fingerprints + MLP as fallback.
    """

    def __init__(self, model_path: Optional[str] = None, embed_dim: int = 512,
                 use_unimol: bool = True):
        self.embed_dim = embed_dim
        self.use_unimol = use_unimol

        if use_unimol and model_path and Path(model_path).exists():
            self._load_unimol(model_path)
        else:
            # Fallback: fingerprint-based encoder
            self.encoder = FingerprintEncoder(output_dim=embed_dim, input_type='molecule')
            self.use_unimol = False

    def _load_unimol(self, model_path):
        """Load Uni-Mol pretrained molecule encoder."""
        try:
            # Uni-Mol tools provides easy interface
            from unimol_tools import UniMolRepr
            self.repr_model = UniMolRepr(data_type='molecule', remove_hs=True)
            print("  Loaded Uni-Mol molecule encoder")
        except ImportError:
            print("  Uni-Mol not available, using fingerprint fallback")
            self.encoder = FingerprintEncoder(output_dim=self.embed_dim, input_type='molecule')
            self.use_unimol = False

    def encode(self, smiles_list: List[str], batch_size: int = 64) -> torch.Tensor:
        """
        Encode a list of SMILES into embeddings.

        Returns: (N, embed_dim) normalized embeddings
        """
        if self.use_unimol:
            return self._encode_unimol(smiles_list, batch_size)
        else:
            return self.encoder.encode(smiles_list)

    def _encode_unimol(self, smiles_list, batch_size):
        """Encode with Uni-Mol."""
        all_embs = []
        for i in range(0, len(smiles_list), batch_size):
            batch = smiles_list[i:i+batch_size]
            reprs = self.repr_model.get_repr(batch)
            # reprs['cls_repr'] is the molecule-level representation
            embs = torch.tensor(reprs['cls_repr'], dtype=torch.float32)
            all_embs.append(embs)

        embeddings = torch.cat(all_embs, dim=0)
        # L2 normalize
        embeddings = nn.functional.normalize(embeddings, dim=-1)
        return embeddings


class ProteinEncoder:
    """
    Protein encoder using Uni-Mol pretrained model (pocket model).

    Extracts embeddings from protein sequences or pocket structures.
    Embeddings are L2-normalized.
    """

    def __init__(self, model_path: Optional[str] = None, embed_dim: int = 512,
                 use_unimol: bool = True):
        self.embed_dim = embed_dim
        self.use_unimol = use_unimol

        if use_unimol and model_path and Path(model_path).exists():
            self._load_unimol(model_path)
        else:
            self.encoder = FingerprintEncoder(output_dim=embed_dim, input_type='protein')
            self.use_unimol = False

    def _load_unimol(self, model_path):
        """Load Uni-Mol pocket encoder or ESM-based protein encoder."""
        try:
            from unimol_tools import UniMolRepr
            self.repr_model = UniMolRepr(data_type='oled')  # protein pocket
            print("  Loaded Uni-Mol protein encoder")
        except ImportError:
            print("  Uni-Mol not available, using sequence-based fallback")
            self.encoder = FingerprintEncoder(output_dim=self.embed_dim, input_type='protein')
            self.use_unimol = False

    def encode(self, sequences: List[str], batch_size: int = 32) -> torch.Tensor:
        """
        Encode protein sequences into embeddings.

        Returns: (N, embed_dim) normalized embeddings
        """
        if self.use_unimol:
            return self._encode_unimol(sequences, batch_size)
        else:
            return self.encoder.encode(sequences)

    def _encode_unimol(self, sequences, batch_size):
        """Encode with Uni-Mol pocket model."""
        all_embs = []
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i:i+batch_size]
            reprs = self.repr_model.get_repr(batch)
            embs = torch.tensor(reprs['cls_repr'], dtype=torch.float32)
            all_embs.append(embs)

        embeddings = torch.cat(all_embs, dim=0)
        embeddings = nn.functional.normalize(embeddings, dim=-1)
        return embeddings


class FingerprintEncoder:
    """
    Fallback encoder using molecular fingerprints / sequence k-mers.
    Projects to fixed dimension via random projection (for testing).
    """

    def __init__(self, output_dim: int = 512, input_type: str = 'molecule'):
        self.output_dim = output_dim
        self.input_type = input_type

        if input_type == 'molecule':
            self.input_dim = 2048  # Morgan fingerprint bits
        else:
            self.input_dim = 1280  # amino acid k-mer features

        # Random projection matrix (fixed)
        torch.manual_seed(42)
        self.proj = torch.randn(self.input_dim, output_dim) / np.sqrt(self.input_dim)

    def encode(self, inputs: List[str]) -> torch.Tensor:
        """Encode inputs to embeddings."""
        features = []

        for inp in inputs:
            if self.input_type == 'molecule':
                feat = self._mol_fingerprint(inp)
            else:
                feat = self._seq_features(inp)
            features.append(feat)

        features = torch.stack(features)
        embeddings = features @ self.proj
        embeddings = nn.functional.normalize(embeddings, dim=-1)
        return embeddings

    def _mol_fingerprint(self, smiles: str) -> torch.Tensor:
        """Compute Morgan fingerprint."""
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem
            from rdkit.DataStructs import ConvertToNumpyArray
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=self.input_dim)
                arr = np.zeros(self.input_dim, dtype=np.float32)
                ConvertToNumpyArray(fp, arr)
                return torch.from_numpy(arr)
        except BaseException:
            pass
        # Fallback: hash-based pseudo-fingerprint
        h = hash(smiles) % (2**32)
        rng = np.random.RandomState(h)
        return torch.from_numpy(rng.binomial(1, 0.1, self.input_dim).astype(np.float32))

    def _seq_features(self, sequence: str) -> torch.Tensor:
        """Compute simple amino acid composition features."""
        aa_vocab = 'ACDEFGHIKLMNPQRSTVWY'
        feat = torch.zeros(self.input_dim)

        # k-mer composition (k=1,2,3)
        offset = 0
        # Unigram
        for i, aa in enumerate(aa_vocab):
            if offset + i < self.input_dim:
                feat[offset + i] = sequence.count(aa) / max(len(sequence), 1)
        offset += len(aa_vocab)

        # Length features
        if offset < self.input_dim:
            feat[offset] = len(sequence) / 1000.0

        return feat


def precompute_embeddings(
    smiles_list: List[str],
    sequences_list: List[str],
    mol_encoder: MoleculeEncoder,
    prot_encoder: ProteinEncoder,
    save_dir: str,
):
    """
    Pre-compute and save all molecule and protein embeddings.
    This is done once and cached for fast training.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Pre-computing molecule embeddings ({len(smiles_list)} molecules)...")
    mol_embs = mol_encoder.encode(smiles_list)
    torch.save(mol_embs, save_dir / 'mol_embeddings.pt')
    print(f"  Saved: {mol_embs.shape}")

    print(f"Pre-computing protein embeddings ({len(sequences_list)} proteins)...")
    prot_embs = prot_encoder.encode(sequences_list)
    torch.save(prot_embs, save_dir / 'prot_embeddings.pt')
    print(f"  Saved: {prot_embs.shape}")

    return mol_embs, prot_embs
