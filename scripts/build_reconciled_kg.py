"""Rebuild base KG with NCBI<->UniProt reconciliation.

Same sources as build_mpi_kg (KEGG + ENZYME + GO + PrimeKG drug_protein), but the
PrimeKG protein endpoint (NCBI gene id) is remapped to UNIPROT:<acc> using the
UniProt idmapping. This makes MPI proteins coincide with the GO/ENZYME/PFam
UniProt proteins, so the MPI subgraph connects to the context graph.

Run merge_pfam.py afterwards to append PFam, then preprocess_kg.py.
"""
import sys, time, gzip
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent.parent))
import scripts.download_kg_sources as D
from data.build_kg import KnowledgeGraph

RAW = Path("data/raw")
t0 = time.time()

# gene -> uniprot acc
print("building gene2uni...", flush=True)
gene2uni = {}
with gzip.open(RAW / "idmap" / "HUMAN_9606_idmapping.dat.gz", "rt") as f:
    for line in f:
        acc, typ, val = line.rstrip("\n").split("\t")
        if typ == "GeneID" and val not in gene2uni:
            gene2uni[val] = acc
print(f"  gene2uni: {len(gene2uni)}", flush=True)

kg = KnowledgeGraph()
kg.add_relation("pseudo_interaction")
print("KEGG...", flush=True);   print("  ->", D.parse_kegg(RAW / "kegg", kg), flush=True)
print("ENZYME...", flush=True); print("  ->", D.parse_enzyme(RAW / "enzyme", kg), flush=True)
print("GO...", flush=True);     print("  ->", D.parse_go_annotations(RAW / "go", kg), flush=True)

print("PrimeKG drug_protein (reconciled)...", flush=True)
n_mpi = n_recon = 0
tmap = {"drug": "molecule", "gene/protein": "protein"}
for ch in pd.read_csv(RAW / "primekg" / "kg.csv", chunksize=500000, low_memory=False):
    d = ch[ch["relation"] == "drug_protein"]
    for _, r in d.iterrows():
        xt, yt = tmap.get(r["x_type"], r["x_type"]), tmap.get(r["y_type"], r["y_type"])
        def ent(idv, t):
            global n_recon
            idv = str(idv)
            if t == "molecule":
                return f"DRUGBANK:{idv}"
            acc = gene2uni.get(idv)              # NCBI gene -> UniProt
            if acc:
                n_recon += 1
                return f"UNIPROT:{acc}"
            return f"NCBI:{idv}"
        kg.add_triple(ent(r["x_id"], xt), "drug_protein", ent(r["y_id"], yt), xt, yt)
        n_mpi += 1
print(f"  -> {n_mpi} MPI triples ({n_recon} endpoints reconciled to UniProt)", flush=True)

kg.summary()
kg.save("data/kg")
print(f"DONE base in {time.time()-t0:.0f}s. triples={kg.num_triples} "
      f"mol={len(kg.molecule_ids)} prot={len(kg.protein_ids)}", flush=True)
