#!/bin/bash
# Download every raw data source for KGOT with working URLs (as of 2026).
# Files land under data/raw/. Total ~8 GB; needs a few GB free after extraction.
# Run from the KGOT project root:  bash scripts/download_all.sh
set -e
UA="Mozilla/5.0"
mkdir -p data/raw/{kegg,enzyme,go,primekg,pfam,idmap,pubchem}
get(){ # get <url> <out>
  if [ -s "$2" ]; then echo "  exists: $2"; return; fi
  echo "  -> $2"; curl -fSL -A "$UA" -o "$2" "$1"
}

echo "== KEGG REST links =="
for pair in \
  "pathway/hsa:pathway_gene" "enzyme/compound:compound_enzyme" \
  "pathway/compound:pathway_compound" "module/pathway:pathway_module" \
  "ko/enzyme:ko_enzyme" "ko/pathway:ko_pathway"; do
  url="https://rest.kegg.jp/link/${pair%%:*}"; out="data/raw/kegg/${pair##*:}.tsv"
  get "$url" "$out"
done

echo "== ENZYME (ExPASy) =="
get "https://ftp.expasy.org/databases/enzyme/enzyme.dat" data/raw/enzyme/enzyme.dat
get "https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/docs/enzclass.txt" data/raw/enzyme/enzclass.txt

echo "== Gene Ontology (human GAF) =="
get "https://current.geneontology.org/annotations/goa_human.gaf.gz" data/raw/go/goa_human.gaf.gz

echo "== PrimeKG (Harvard Dataverse kg.csv) =="
get "https://dataverse.harvard.edu/api/access/datafile/6180620" data/raw/primekg/kg.csv

echo "== PFam regions (UniProt<->family) =="
get "https://ftp.ebi.ac.uk/pub/databases/Pfam/releases/Pfam37.4/Pfam-A.regions.tsv.gz" data/raw/pfam/Pfam-A.regions.tsv.gz

echo "== UniProt idmapping (human; GeneID<->UniProt) =="
get "https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/idmapping/by_organism/HUMAN_9606_idmapping.dat.gz" data/raw/idmap/HUMAN_9606_idmapping.dat.gz

echo "== PubChem FTP bulk (DrugBank->CID->SMILES; not rate-limited) =="
get "https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras/CID-Synonym-filtered.gz" data/raw/pubchem/CID-Synonym-filtered.gz
get "https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras/CID-SMILES.gz" data/raw/pubchem/CID-SMILES.gz

echo "All raw sources downloaded to data/raw/."
