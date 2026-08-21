"""
Molecular Similarity Computation.

Computes Tanimoto similarity matrix between molecules using fingerprints.
Used for the similarity constraint in OT pseudo-labeling (Eq. 7).
"""

import numpy as np
from typing import List, Optional

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False


def compute_fingerprint(smiles: str, radius: int = 2, n_bits: int = 2048):
    """Compute Morgan fingerprint for a SMILES string."""
    if not HAS_RDKIT:
        raise RuntimeError("RDKit required for fingerprint computation")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)


def tanimoto_similarity_matrix(smiles_list: List[str], n_bits: int = 2048) -> np.ndarray:
    """
    Compute pairwise Tanimoto similarity matrix.

    Args:
        smiles_list: list of M SMILES strings

    Returns:
        Sim: (M, M) numpy array of Tanimoto similarities
    """
    fps = []
    for smi in smiles_list:
        fp = compute_fingerprint(smi, n_bits=n_bits)
        if fp is None:
            # Fallback: zero fingerprint
            fp = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles('C'), 2, nBits=n_bits)
        fps.append(fp)

    M = len(fps)
    Sim = np.zeros((M, M), dtype=np.float32)

    for i in range(M):
        for j in range(i, M):
            sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            Sim[i, j] = sim
            Sim[j, i] = sim

    return Sim


def batch_tanimoto_from_fingerprints(fp_array: np.ndarray) -> np.ndarray:
    """
    Fast Tanimoto from pre-computed binary fingerprint arrays.

    Args:
        fp_array: (M, n_bits) binary numpy array

    Returns:
        Sim: (M, M) Tanimoto similarity matrix
    """
    # Tanimoto = |A∩B| / |A∪B| = dot(A,B) / (|A| + |B| - dot(A,B))
    fp = fp_array.astype(np.float32)
    dot = fp @ fp.T
    norms = np.diag(dot)  # |A| = sum(A) for binary vectors

    denom = norms[:, None] + norms[None, :] - dot
    Sim = np.where(denom > 0, dot / denom, 0.0)

    return Sim.astype(np.float32)
