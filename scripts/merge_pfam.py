"""Append PFam protein->family edges to the EXISTING KG (preserving all current
entity IDs so previously-computed MPI id lists / embeddings stay valid).

Streams Pfam-A.regions.tsv.gz (114M region rows), dedupes to unique
(uniprot_acc, pfamA_acc) pairs -> ~protein->family edges, and adds them on top
of the existing KG. New UNIPROT proteins expand the protein candidate pool
(needed to reproduce the aligned eval's ~6.45M-protein pool).
"""
import sys, time, gzip
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from data.build_kg import KnowledgeGraph

t0 = time.time()
print("loading existing KG...", flush=True)
kg = KnowledgeGraph.load("data/kg")
print(f"  before: entities={kg.num_entities} triples={kg.num_triples} "
      f"prot={len(kg.protein_ids)}", flush=True)

REL = "belongs_to_family"
kg.add_relation(REL)
rel_id = kg.relation2id[REL]

seen = set()          # unique (prot, fam) pairs already added
path = "data/raw/pfam/Pfam-A.regions.tsv.gz"
MAX_EDGES = 9_100_000  # cap to match the original's ~9.1M PFam edges / documented 10.9M total
n_rows = n_edges = 0
with gzip.open(path, "rt") as f:
    header = f.readline()  # skip header
    for line in f:
        n_rows += 1
        p = line.split("\t", 5)
        if len(p) < 5:
            continue
        prot, fam = p[0], p[4]
        key = prot + "\t" + fam
        if key in seen:
            continue
        seen.add(key)
        kg.add_triple(f"UNIPROT:{prot}", REL, f"PFAM:{fam}", "protein", "protein_family")
        n_edges += 1
        if n_edges >= MAX_EDGES:
            print(f"  reached cap {MAX_EDGES} at {n_rows} rows", flush=True)
            break
        if n_rows % 10_000_000 == 0:
            print(f"  {n_rows/1e6:.0f}M rows, {n_edges} unique edges, "
                  f"{time.time()-t0:.0f}s", flush=True)

print(f"parsed {n_rows} rows -> {n_edges} unique protein-family edges", flush=True)
kg.summary()
kg.save("data/kg")
print(f"DONE in {time.time()-t0:.0f}s. entities={kg.num_entities} "
      f"triples={kg.num_triples} prot={len(kg.protein_ids)}", flush=True)
