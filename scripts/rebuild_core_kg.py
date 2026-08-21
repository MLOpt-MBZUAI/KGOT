"""Phase 1 KG rebuild: download + parse the 4 reliable core sources
(ChEBI, KEGG, GO, ENZYME) and save the KG. PFam/DrugBank are handled in a
later phase because they are large and need special handling. This verifies the
whole build path works on the new machine end to end."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import scripts.download_kg_sources as D
from data.build_kg import KnowledgeGraph

t0 = time.time()
print("=== Phase 1: downloading core sources ===", flush=True)
chebi_dir = D.download_chebi()
kegg_dir = D.download_kegg()
go_dir = D.download_gene_ontology()
enzyme_dir = D.download_enzyme()

print("\n=== Building core KG ===", flush=True)
kg = KnowledgeGraph()
kg.add_relation('pseudo_interaction')
print("ChEBI...", flush=True);  print("  ->", D.parse_chebi_relations(chebi_dir, kg), flush=True)
print("KEGG...", flush=True);   print("  ->", D.parse_kegg(kegg_dir, kg), flush=True)
print("GO...", flush=True);     print("  ->", D.parse_go_annotations(go_dir, kg), flush=True)
print("ENZYME...", flush=True); print("  ->", D.parse_enzyme(enzyme_dir, kg), flush=True)

kg.summary()
kg.save("data/kg")
print(f"\nDONE core KG in {time.time()-t0:.0f}s. Triples: {kg.num_triples}", flush=True)
