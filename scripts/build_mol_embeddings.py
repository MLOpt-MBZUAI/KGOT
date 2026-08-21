"""Molecule embeddings from REAL SMILES (no hash fallback).

1. For each MPI drug: fetch canonical SMILES from PubChem (by name, then by
   DrugBank registry id). Results cached to data/embeddings/smiles_cache.pt so
   the run is resumable.
2. Morgan fingerprint (radius=2, 2048 bits) for molecules with valid SMILES.
3. Fixed-seed Gaussian random projection 2048 -> 512, L2-normalized.
   Drugs without SMILES get a zero vector (excluded downstream, like the original
   ~67% coverage).

Outputs: data/embeddings/{mol_embeddings.pt, mol_kg_ids.npy, drug_smiles_real.npy}
"""
import torch, numpy as np, time, json, urllib.request, urllib.parse, sys
from pathlib import Path
from tqdm import tqdm

OUT = Path("data/embeddings"); OUT.mkdir(parents=True, exist_ok=True)
CACHE = OUT / "smiles_cache.pt"

mol_ids = np.load("data/kg/mpi_molecule_ids.npy")
drug_map = torch.load("data/kg/mpi_drug_map.pt", weights_only=False)

cache = torch.load(CACHE, weights_only=False) if CACHE.exists() else {}
print(f"{len(mol_ids)} drugs; cache has {len(cache)} entries", flush=True)

UA = {"User-Agent": "Mozilla/5.0"}
_last = [0.0]
def _get(url, retries=5):
    for attempt in range(retries + 1):
        dt = time.time() - _last[0]
        if dt < 0.4:             # gentle ~2.5 req/s to avoid ServerBusy blocks
            time.sleep(0.4 - dt)
        _last[0] = time.time()
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=12) as r:
                data = json.loads(r.read())
            if isinstance(data, dict) and "Fault" in data:
                raise RuntimeError("ServerBusy")  # PubChem 200-with-Fault
            return data
        except Exception as e:
            s = str(e)
            if "503" in s or "429" in s or "ServerBusy" in s or "Busy" in s:
                time.sleep(min(3.0 * (attempt + 1), 30.0))  # progressive backoff
                continue
            return None
    return None

SMI_KEYS = ("SMILES", "IsomericSMILES", "ConnectivitySMILES", "CanonicalSMILES")
def _extract(d):
    try:
        props = d["PropertyTable"]["Properties"][0]
        for k in SMI_KEYS:
            if props.get(k):
                return props[k]
    except Exception:
        pass
    return None

def fetch_smiles(name, dbid):
    # 1) by drug name
    if name and name != "nan":
        d = _get(f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
                 f"{urllib.parse.quote(name)}/property/SMILES/JSON")
        s = _extract(d) if d else None
        if s:
            return s
    # 2) by DrugBank registry id -> CID -> SMILES (higher coverage, like the original)
    if dbid and dbid.startswith("DB"):
        d = _get(f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/xref/"
                 f"RegistryID/{urllib.parse.quote(dbid)}/cids/JSON")
        try:
            cid = d["IdentifierList"]["CID"][0]
        except Exception:
            cid = None
        if cid:
            d2 = _get(f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/"
                      f"{cid}/property/SMILES/JSON")
            s = _extract(d2) if d2 else None
            if s:
                return s
    return None

# ---- Stage 1: fetch SMILES (resumable; retry entries that previously failed) ----
todo = [int(m) for m in mol_ids if not cache.get(str(int(m)))]
print(f"fetching {len(todo)} missing SMILES...", flush=True)
for i, kid in enumerate(tqdm(todo, desc="PubChem")):
    dbid, name = drug_map[kid]
    cache[str(kid)] = fetch_smiles(name, dbid)  # may be None
    if (i + 1) % 100 == 0:
        torch.save(cache, CACHE)
        time.sleep(0.1)
torch.save(cache, CACHE)
got = sum(1 for m in mol_ids if cache.get(str(int(m))))
print(f"SMILES coverage: {got}/{len(mol_ids)} ({100*got/len(mol_ids):.1f}%)", flush=True)

# ---- Stage 2: Morgan fingerprints -> 512 projection ----
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

rng = np.random.RandomState(0)
proj = rng.randn(2048, 512).astype(np.float32) / np.sqrt(2048)

smiles_out, fps = [], np.zeros((len(mol_ids), 2048), dtype=np.float32)
n_valid = 0
for i, m in enumerate(mol_ids):
    smi = cache.get(str(int(m)))
    smiles_out.append(smi or "")
    if not smi:
        continue
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        continue
    bv = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    arr = np.zeros((2048,), dtype=np.float32)
    Chem.DataStructs.ConvertToNumpyArray(bv, arr)
    fps[i] = arr
    n_valid += 1

emb = torch.tensor(fps @ proj, dtype=torch.float32)
emb = torch.nn.functional.normalize(emb, dim=-1)
torch.save(emb, OUT / "mol_embeddings.pt")
np.save(OUT / "mol_kg_ids.npy", mol_ids.astype(np.int64))
np.save(OUT / "drug_smiles_real.npy", np.array(smiles_out, dtype=object))
print(f"valid Morgan fps: {n_valid}/{len(mol_ids)}; mol_embeddings {tuple(emb.shape)}", flush=True)
print("DONE", flush=True)
