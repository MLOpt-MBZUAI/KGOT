"""
Download and Process All 6 KG Data Sources for KGOT.

Sources (from paper Appendix B):
1. CHEBI + UniProt: 1.35M catalytic activity relations (30K molecules, 500K proteins)
2. KEGG: Genes, pathways, modules, compounds, KO
3. Gene Ontology: Biological process / cellular component / molecular function
4. PFam: Protein family classifications
5. ENZYME (EC): Enzyme commission numbers, enzyme-substrate relationships
6. DrugBank: Drug-target interactions (supplementary)

Final KG: 8 node types, 29 edge types, 6,483,852 relationships.

Entity types and counts (Table 5):
- UNIPROT: 5,956,325 (head)
- CHEBI: 336,374 (head), 1,678,407 (tail)
- KEGG: 92,184 (head)
- GO: 89,235 (head), 3,191,321 (tail)
- EC: 8,459 (head), 304,428 (tail)
- PFAM: 792,235 (tail)
- KEGG KO: 407,307 (tail)
- KPATHWAY: 89,989 (tail)
- KMODULE: 1,275 (head), 3,470 (tail)
- KCOMPOUND: 16,695 (head)
"""

import os
import sys
import gzip
import urllib.request
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.build_kg import KnowledgeGraph


RAW_DIR = Path('./data/raw')
KG_DIR = Path('./data/kg')


def download_file(url, output_path, desc=""):
    """Download a file with progress."""
    output_path = Path(output_path)
    if output_path.exists():
        print(f"  Already exists: {output_path}")
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading {desc}: {url}")

    try:
        urllib.request.urlretrieve(url, str(output_path))
        print(f"  → Saved: {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)")
    except Exception as e:
        print(f"  ✗ Failed: {e}")

    return output_path


# ============================================================
# Source 1: ChEBI (molecules) + UniProt (proteins)
# ChEBI provides compound classifications and relationships
# UniProt provides protein-enzyme catalytic activities
# ============================================================

def download_chebi():
    """
    Download ChEBI compound data.
    ChEBI: Chemical Entities of Biological Interest.
    We need: chebi_id, name, and relationships to proteins.
    """
    print("\n=== Source 1: ChEBI ===")
    chebi_dir = RAW_DIR / 'chebi'
    chebi_dir.mkdir(parents=True, exist_ok=True)

    # ChEBI compounds (TSV format)
    # Contains compound IDs, names, and ontology relationships
    download_file(
        "https://ftp.ebi.ac.uk/pub/databases/chebi/Flat_file_tab_delimited/compounds.tsv.gz",
        chebi_dir / 'compounds.tsv.gz',
        "ChEBI compounds"
    )

    # ChEBI relations (parent-child, has_role, etc.)
    download_file(
        "https://ftp.ebi.ac.uk/pub/databases/chebi/Flat_file_tab_delimited/relation.tsv",
        chebi_dir / 'relation.tsv',
        "ChEBI relations"
    )

    return chebi_dir


def download_uniprot_enzymatic():
    """
    Download UniProt enzyme-substrate relationships.
    Links proteins (UniProt IDs) to substrates (ChEBI IDs) via catalytic activity.
    """
    print("\n=== Source 1b: UniProt (enzyme-substrate) ===")
    uniprot_dir = RAW_DIR / 'uniprot'
    uniprot_dir.mkdir(parents=True, exist_ok=True)

    # UniProt ID mapping (to EC numbers, to ChEBI)
    # The full UniProt SPARQL or flat file is huge; we use the enzymatic subset
    # Swiss-Prot (reviewed) enzyme entries
    download_file(
        "https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.dat.gz",
        uniprot_dir / 'uniprot_sprot.dat.gz',
        "UniProt Swiss-Prot (reviewed, ~570MB compressed)"
    )

    return uniprot_dir


# ============================================================
# Source 2: KEGG (pathways, genes, modules, compounds)
# ============================================================

def download_kegg():
    """
    Download KEGG pathway and gene data.
    KEGG REST API provides: pathway-gene, compound-enzyme links.

    Note: KEGG requires academic license for bulk download.
    We use the free REST API for smaller subsets.
    """
    print("\n=== Source 2: KEGG ===")
    kegg_dir = RAW_DIR / 'kegg'
    kegg_dir.mkdir(parents=True, exist_ok=True)

    # KEGG links are available via REST API
    # http://rest.kegg.jp/link/{target_db}/{source_db}
    kegg_links = {
        'pathway_gene': 'http://rest.kegg.jp/link/pathway/hsa',      # human gene → pathway
        'compound_enzyme': 'http://rest.kegg.jp/link/enzyme/compound', # compound → enzyme
        'pathway_compound': 'http://rest.kegg.jp/link/pathway/compound',
        'pathway_module': 'http://rest.kegg.jp/link/module/pathway',
        'ko_enzyme': 'http://rest.kegg.jp/link/ko/enzyme',           # KO → enzyme
        'ko_pathway': 'http://rest.kegg.jp/link/ko/pathway',         # KO → pathway
    }

    for name, url in kegg_links.items():
        download_file(url, kegg_dir / f'{name}.tsv', f"KEGG {name}")

    return kegg_dir


# ============================================================
# Source 3: Gene Ontology
# ============================================================

def download_gene_ontology():
    """
    Download Gene Ontology annotations.
    GO annotations link proteins to biological processes, cellular components,
    and molecular functions.
    """
    print("\n=== Source 3: Gene Ontology ===")
    go_dir = RAW_DIR / 'go'
    go_dir.mkdir(parents=True, exist_ok=True)

    # GO annotation file (GAF format) - human
    download_file(
        "http://current.geneontology.org/annotations/goa_human.gaf.gz",
        go_dir / 'goa_human.gaf.gz',
        "GO annotations (human)"
    )

    # GO ontology structure (OBO format)
    download_file(
        "http://purl.obolibrary.org/obo/go/go-basic.obo",
        go_dir / 'go-basic.obo',
        "GO ontology structure"
    )

    return go_dir


# ============================================================
# Source 4: PFam (protein families)
# ============================================================

def download_pfam():
    """
    Download PFam protein family annotations.
    Maps UniProt proteins to PFam family IDs.
    """
    print("\n=== Source 4: PFam ===")
    pfam_dir = RAW_DIR / 'pfam'
    pfam_dir.mkdir(parents=True, exist_ok=True)

    # Pfam-A full (protein → family mapping)
    # InterPro provides a lighter version
    download_file(
        "https://ftp.ebi.ac.uk/pub/databases/Pfam/current_release/Pfam-A.full.uniprot.gz",
        pfam_dir / 'Pfam-A.full.uniprot.gz',
        "PFam-A UniProt annotations (large, ~3GB)"
    )

    # Alternative: lighter mapping file
    download_file(
        "https://ftp.ebi.ac.uk/pub/databases/Pfam/current_release/database_files/pfamA_reg_full_significant.txt.gz",
        pfam_dir / 'pfam_reg_full.txt.gz',
        "PFam region annotations"
    )

    return pfam_dir


# ============================================================
# Source 5: ENZYME (EC numbers)
# ============================================================

def download_enzyme():
    """
    Download ENZYME database (EC numbers).
    Maps enzyme commission numbers to proteins.
    """
    print("\n=== Source 5: ENZYME ===")
    enzyme_dir = RAW_DIR / 'enzyme'
    enzyme_dir.mkdir(parents=True, exist_ok=True)

    # ENZYME flat file
    download_file(
        "https://ftp.expasy.org/databases/enzyme/enzyme.dat",
        enzyme_dir / 'enzyme.dat',
        "ENZYME database"
    )

    # EC → UniProt mapping (from UniProt)
    download_file(
        "https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/docs/enzclass.txt",
        enzyme_dir / 'enzclass.txt',
        "EC classification"
    )

    return enzyme_dir


# ============================================================
# Source 6: DrugBank (optional, needs registration)
# ============================================================

def download_drugbank():
    """
    DrugBank drug-target interactions.
    Note: DrugBank requires registration for full download.
    We provide instructions; the user needs to obtain the file manually.
    """
    print("\n=== Source 6: DrugBank ===")
    db_dir = RAW_DIR / 'drugbank'
    db_dir.mkdir(parents=True, exist_ok=True)

    readme = db_dir / 'README.txt'
    if not readme.exists():
        with open(readme, 'w') as f:
            f.write("DrugBank Drug-Target Interactions\n")
            f.write("================================\n\n")
            f.write("DrugBank requires registration for download.\n")
            f.write("1. Register at: https://go.drugbank.com/\n")
            f.write("2. Download: 'All Drug Target Identifiers' (CSV)\n")
            f.write("3. Place as: drugbank_targets.csv\n")
            f.write("   Columns needed: drugbank_id, uniprot_id, action\n")

    print("  DrugBank requires manual download (registration needed)")
    print("  See: data/raw/drugbank/README.txt")

    return db_dir


# ============================================================
# Build KG from downloaded sources
# ============================================================

def parse_chebi_relations(chebi_dir: Path, kg: KnowledgeGraph) -> int:
    """Parse ChEBI compound relations."""
    count = 0
    rel_file = chebi_dir / 'relation.tsv'

    if not rel_file.exists():
        return 0

    df = pd.read_csv(rel_file, sep='\t', header=0, on_bad_lines='skip')
    print(f"  ChEBI relations file: {len(df)} rows, columns: {list(df.columns)[:5]}")

    for _, row in tqdm(df.iterrows(), total=len(df), desc="  ChEBI rels"):
        try:
            head = f"CHEBI:{row.iloc[1]}"
            rel = str(row.iloc[2]) if len(row) > 2 else 'chebi_relation'
            tail = f"CHEBI:{row.iloc[3]}" if len(row) > 3 else f"CHEBI:{row.iloc[0]}"
            kg.add_triple(head, f"chebi_{rel}", tail, 'molecule', 'molecule')
            count += 1
        except:
            continue

    return count


def parse_kegg(kegg_dir: Path, kg: KnowledgeGraph) -> int:
    """Parse KEGG link files."""
    count = 0

    for tsv_file in kegg_dir.glob('*.tsv'):
        if tsv_file.stat().st_size == 0:
            continue

        rel_name = tsv_file.stem
        try:
            with open(tsv_file) as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        head = parts[0].strip()
                        tail = parts[1].strip()
                        kg.add_triple(head, f"kegg_{rel_name}", tail, 'kegg_entity', 'kegg_entity')
                        count += 1
        except:
            continue

    return count


def parse_go_annotations(go_dir: Path, kg: KnowledgeGraph) -> int:
    """Parse Gene Ontology annotations (GAF format)."""
    count = 0
    gaf_file = go_dir / 'goa_human.gaf.gz'

    if not gaf_file.exists():
        return 0

    import gzip
    with gzip.open(gaf_file, 'rt') as f:
        for line in tqdm(f, desc="  GO annotations"):
            if line.startswith('!'):
                continue
            parts = line.strip().split('\t')
            if len(parts) < 7:
                continue

            protein_id = parts[1]  # UniProt accession
            go_id = parts[4]       # GO:XXXXXXX
            aspect = parts[8]      # P=biological_process, C=cellular_component, F=molecular_function

            aspect_map = {'P': 'biological_process', 'C': 'cellular_component', 'F': 'molecular_function'}
            rel = f"go_{aspect_map.get(aspect, aspect)}"

            kg.add_triple(f"UNIPROT:{protein_id}", rel, go_id, 'protein', 'go_term')
            count += 1

    return count


def parse_enzyme(enzyme_dir: Path, kg: KnowledgeGraph) -> int:
    """Parse ENZYME database."""
    count = 0
    dat_file = enzyme_dir / 'enzyme.dat'

    if not dat_file.exists():
        return 0

    current_ec = None
    with open(dat_file) as f:
        for line in f:
            if line.startswith('ID'):
                current_ec = line.strip().split(None, 1)[1].strip()
            elif line.startswith('DR') and current_ec:
                # DR lines contain UniProt links: "DR   P12345, NAME;  P67890, NAME;"
                entries = line[5:].strip().rstrip(';').split(';')
                for entry in entries:
                    entry = entry.strip()
                    if ',' in entry:
                        uniprot_id = entry.split(',')[0].strip()
                        if uniprot_id:
                            kg.add_triple(f"EC:{current_ec}", 'catalyzed_by', f"UNIPROT:{uniprot_id}", 'ec_number', 'protein')
                            count += 1
            elif line.startswith('//'):
                current_ec = None

    return count


def main():
    """Download all sources and build KG."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    KG_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("KGOT: Downloading KG Data Sources")
    print("=" * 60)

    # Download all sources
    chebi_dir = download_chebi()
    # uniprot_dir = download_uniprot_enzymatic()  # Very large (570MB), skip for now
    kegg_dir = download_kegg()
    go_dir = download_gene_ontology()
    pfam_dir = download_pfam()
    enzyme_dir = download_enzyme()
    download_drugbank()

    # Build KG
    print("\n" + "=" * 60)
    print("Building Knowledge Graph")
    print("=" * 60)

    kg = KnowledgeGraph()
    kg.add_relation('pseudo_interaction')  # Reserve for later

    print("\nParsing ChEBI...")
    n1 = parse_chebi_relations(chebi_dir, kg)
    print(f"  → {n1} triples")

    print("\nParsing KEGG...")
    n2 = parse_kegg(kegg_dir, kg)
    print(f"  → {n2} triples")

    print("\nParsing Gene Ontology...")
    n3 = parse_go_annotations(go_dir, kg)
    print(f"  → {n3} triples")

    print("\nParsing ENZYME...")
    n4 = parse_enzyme(enzyme_dir, kg)
    print(f"  → {n4} triples")

    print("\n" + "=" * 60)
    kg.summary()

    # Save
    kg.save(str(KG_DIR))

    print("\n" + "=" * 60)
    print("KG Construction Complete!")
    print(f"Total triples: {kg.num_triples}")
    print(f"Saved to: {KG_DIR}")
    print("=" * 60)


if __name__ == '__main__':
    main()
