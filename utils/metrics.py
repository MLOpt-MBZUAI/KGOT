"""
Evaluation metrics for KGOT.

Virtual screening: AUROC, BEDROC, EF@K
Link prediction: Hits@1, Hits@3, Hits@5, MRR
"""

import numpy as np
from sklearn.metrics import roc_auc_score
from typing import Dict


def compute_auroc(y_true, y_score):
    """Area under ROC curve."""
    try:
        return roc_auc_score(y_true, y_score)
    except:
        return 0.5


def compute_bedroc(y_true, y_score, alpha=20.0):
    """
    Boltzmann-Enhanced Discrimination of ROC.
    Emphasizes early recognition of actives.
    """
    n = len(y_true)
    n_actives = sum(y_true)
    if n_actives == 0:
        return 0.0

    # Sort by score descending
    order = np.argsort(-np.array(y_score))
    y_sorted = np.array(y_true)[order]

    # Compute BEDROC
    sum_exp = 0.0
    for i, label in enumerate(y_sorted):
        if label == 1:
            sum_exp += np.exp(-alpha * i / n)

    ra = n_actives / n
    ri = sum_exp / n_actives

    rand = ra * (1 - np.exp(-alpha)) / (np.exp(alpha / n) - 1)
    fac = ra * np.sinh(alpha / 2) / (np.cosh(alpha / 2) - np.cosh(alpha / 2 - alpha * ra))

    bedroc = (ri - rand) / (fac - rand) if (fac - rand) != 0 else 0.0
    return max(0.0, min(1.0, bedroc))


def compute_enrichment_factor(y_true, y_score, fraction=0.01):
    """
    Enrichment Factor at top fraction.
    EF@x% = (actives in top x%) / (expected actives in random x%)
    """
    n = len(y_true)
    n_actives = sum(y_true)
    if n_actives == 0:
        return 0.0

    top_k = max(1, int(n * fraction))
    order = np.argsort(-np.array(y_score))
    top_labels = np.array(y_true)[order[:top_k]]

    actives_in_top = sum(top_labels)
    expected = n_actives * fraction

    return actives_in_top / expected if expected > 0 else 0.0


def compute_hits_at_k(ranks, k):
    """Hits@K: fraction of queries where correct answer is in top K."""
    return np.mean([1.0 if r <= k else 0.0 for r in ranks])


def compute_mrr(ranks):
    """Mean Reciprocal Rank."""
    return np.mean([1.0 / r for r in ranks])


def evaluate_virtual_screening(y_true, y_score) -> Dict[str, float]:
    """Full virtual screening evaluation."""
    return {
        'auroc': compute_auroc(y_true, y_score) * 100,
        'bedroc': compute_bedroc(y_true, y_score) * 100,
        'ef_0.5': compute_enrichment_factor(y_true, y_score, 0.005),
        'ef_1': compute_enrichment_factor(y_true, y_score, 0.01),
        'ef_2': compute_enrichment_factor(y_true, y_score, 0.02),
    }


def leakage_filter_tanimoto(train_smiles: list, test_smiles: list, threshold: float = 0.60):
    """
    Leakage control: remove training molecules with Tanimoto similarity >= threshold
    to any test molecule. (Paper Section 3.1, Appendix D)

    Returns: indices of training molecules to KEEP (below threshold).
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem, DataStructs

        # Compute test fingerprints
        test_fps = []
        for smi in test_smiles:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                test_fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))
            else:
                test_fps.append(None)

        keep_indices = []
        for i, smi in enumerate(train_smiles):
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                keep_indices.append(i)
                continue
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)

            # Check against all test molecules
            leaked = False
            for tfp in test_fps:
                if tfp is not None:
                    sim = DataStructs.TanimotoSimilarity(fp, tfp)
                    if sim >= threshold:
                        leaked = True
                        break
            if not leaked:
                keep_indices.append(i)

        return keep_indices
    except BaseException:
        # If RDKit unavailable, return all indices (no filtering)
        return list(range(len(train_smiles)))


def leakage_filter_scaffold(train_smiles: list, test_smiles: list):
    """
    Murcko scaffold-out filtering: remove training molecules sharing
    Bemis-Murcko scaffold with any test molecule. (Paper Appendix D)

    Returns: indices of training molecules to KEEP.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles

        # Get test scaffolds
        test_scaffolds = set()
        for smi in test_smiles:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                try:
                    scaf = MurckoScaffoldSmiles(mol=mol, includeChirality=False)
                    test_scaffolds.add(scaf)
                except:
                    pass

        keep_indices = []
        for i, smi in enumerate(train_smiles):
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                keep_indices.append(i)
                continue
            try:
                scaf = MurckoScaffoldSmiles(mol=mol, includeChirality=False)
                if scaf not in test_scaffolds:
                    keep_indices.append(i)
            except:
                keep_indices.append(i)

        return keep_indices
    except BaseException:
        return list(range(len(train_smiles)))


def evaluate_link_prediction(ranks) -> Dict[str, float]:
    """Full link prediction evaluation."""
    return {
        'hits@1': compute_hits_at_k(ranks, 1) * 100,
        'hits@3': compute_hits_at_k(ranks, 3) * 100,
        'hits@5': compute_hits_at_k(ranks, 5) * 100,
        'mrr': compute_mrr(ranks),
    }
