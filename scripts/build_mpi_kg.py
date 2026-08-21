"""Build the MPI-focused KG: KEGG + ENZYME (already downloaded) + GO + PrimeKG's
drug_protein edges (the 51,306 MPI interactions). PFam/ChEBI added in a later phase.

Drug entities  -> DRUGBANK:<id>  (type 'molecule')
Protein entities-> NCBI:<gene_id> (type 'protein')  [MPI proteins]
This mirrors the original build whose MPI edges came from DrugBank-via-PrimeKG.
"""
import sys, time
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent.parent))
import scripts.download_kg_sources as D
from data.build_kg import KnowledgeGraph

RAW = Path("data/raw")
t0 = time.time()
kg = KnowledgeGraph()
kg.add_relation("pseudo_interaction")

print("KEGG...", flush=True);   print("  ->", D.parse_kegg(RAW / "kegg", kg), flush=True)
print("ENZYME...", flush=True); print("  ->", D.parse_enzyme(RAW / "enzyme", kg), flush=True)
print("GO...", flush=True);     print("  ->", D.parse_go_annotations(RAW / "go", kg), flush=True)

print("PrimeKG drug_protein (MPI edges)...", flush=True)
n_mpi = 0
tmap = {"drug": "molecule", "gene/protein": "protein"}
for ch in pd.read_csv(RAW / "primekg" / "kg.csv", chunksize=500000, low_memory=False):
    d = ch[ch["relation"] == "drug_protein"]
    for _, r in d.iterrows():
        xt, yt = tmap.get(r["x_type"], r["x_type"]), tmap.get(r["y_type"], r["y_type"])
        # canonical entity ids (prefix by source id space) so both directions share nodes
        xid = f"DRUGBANK:{r['x_id']}" if xt == "molecule" else f"NCBI:{r['x_id']}"
        yid = f"DRUGBANK:{r['y_id']}" if yt == "molecule" else f"NCBI:{r['y_id']}"
        kg.add_triple(xid, "drug_protein", yid, xt, yt)
        n_mpi += 1
print(f"  -> {n_mpi} MPI triples", flush=True)

kg.summary()
kg.save("data/kg")
print(f"\nDONE in {time.time()-t0:.0f}s. Triples={kg.num_triples} "
      f"Mol={len(kg.molecule_ids)} Prot={len(kg.protein_ids)}", flush=True)
