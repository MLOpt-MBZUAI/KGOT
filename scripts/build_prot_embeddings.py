"""Protein embeddings via ESM-2 650M from REAL UniProt sequences (no hash).

1. For each MPI protein (NCBI gene id + symbol): fetch the human (organism 9606)
   canonical sequence from UniProt by exact gene name. Cached + resumable.
2. Encode with ESM-2 t33 650M (mean-pool last hidden state -> 1280).
3. Seeded projection 1280 -> 512, L2-normalized.

Outputs: data/embeddings/{prot_embeddings.pt, prot_kg_ids.npy, prot_sequences.npy}
"""
import torch, numpy as np, time, json, urllib.request, urllib.parse, sys
from pathlib import Path
from tqdm import tqdm

OUT = Path("data/embeddings"); OUT.mkdir(parents=True, exist_ok=True)
SEQ_CACHE = OUT / "seq_cache.pt"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

prot_ids = np.load("data/kg/mpi_protein_ids.npy")
prot_map = torch.load("data/kg/mpi_prot_map.pt", weights_only=False)
cache = torch.load(SEQ_CACHE, weights_only=False) if SEQ_CACHE.exists() else {}
print(f"{len(prot_ids)} proteins; seq cache {len(cache)}", flush=True)

UA = {"User-Agent": "Mozilla/5.0"}
def fetch_seq(symbol):
    if not symbol or symbol == "nan":
        return None
    q = urllib.parse.quote(f"gene_exact:{symbol} AND organism_id:9606 AND reviewed:true")
    url = (f"https://rest.uniprot.org/uniprotkb/search?query={q}"
           f"&fields=sequence&format=json&size=1")
    try:
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=10) as r:
            res = json.loads(r.read()).get("results", [])
            if res:
                seq = res[0].get("sequence", {}).get("value", "")
                return seq if len(seq) > 10 else None
    except Exception:
        return None
    return None

# ---- Stage 1: fetch sequences (resumable) ----
todo = [int(p) for p in prot_ids if str(int(p)) not in cache]
print(f"fetching {len(todo)} sequences from UniProt...", flush=True)
for i, kid in enumerate(tqdm(todo, desc="UniProt")):
    _, sym = prot_map[kid]
    cache[str(kid)] = fetch_seq(sym)
    if (i + 1) % 100 == 0:
        torch.save(cache, SEQ_CACHE); time.sleep(0.1)
torch.save(cache, SEQ_CACHE)
got = sum(1 for p in prot_ids if cache.get(str(int(p))))
print(f"sequence coverage: {got}/{len(prot_ids)} ({100*got/len(prot_ids):.1f}%)", flush=True)

seqs = [cache.get(str(int(p))) for p in prot_ids]
np.save(OUT / "prot_sequences.npy", np.array(seqs, dtype=object))

# ---- Stage 2: ESM-2 650M encoding ----
from transformers import AutoTokenizer, AutoModel
name = "facebook/esm2_t33_650M_UR50D"
print("loading ESM-2 650M...", flush=True)
tok = AutoTokenizer.from_pretrained(name)
model = AutoModel.from_pretrained(name).to(DEVICE).eval()

embs, bs, max_len = [], 8, 1022
for i in tqdm(range(0, len(seqs), bs), desc="ESM-2"):
    batch = [(s[:max_len] if s and len(s) > 10 else "MAAAA") for s in seqs[i:i+bs]]
    inp = tok(batch, return_tensors="pt", padding=True, truncation=True,
              max_length=max_len + 2).to(DEVICE)
    with torch.no_grad():
        out = model(**inp)
        m = inp["attention_mask"].unsqueeze(-1)
        pooled = (out.last_hidden_state * m).sum(1) / m.sum(1)
    embs.append(pooled.float().cpu())
emb = torch.cat(embs, 0)  # (N, 1280)

g = torch.Generator().manual_seed(0)
proj = torch.randn(emb.size(1), 512, generator=g) / np.sqrt(emb.size(1))
emb = torch.nn.functional.normalize(emb @ proj, dim=-1)
torch.save(emb, OUT / "prot_embeddings.pt")
np.save(OUT / "prot_kg_ids.npy", prot_ids.astype(np.int64))
print(f"prot_embeddings {tuple(emb.shape)}; coverage {got}/{len(prot_ids)}", flush=True)
print("DONE", flush=True)
