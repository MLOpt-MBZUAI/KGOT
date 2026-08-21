"""
Knowledge Graph Construction from Biological Datasets.

Integrates 6 data sources into a unified KG:
1. CHEBI → molecule–protein catalytic activity (1.35M relations)
2. KEGG → gene regulation, metabolic pathways
3. Gene Ontology → biological process, cellular component annotations
4. PFam → protein family classifications
5. ENZYME → enzyme-substrate relationships
6. DrugBank → drug-target interactions

Final KG: >3M relations, entities include molecules, proteins, genes, pathways, GO terms.

Reference: Section 2.1, Appendix B.
"""

import os
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from tqdm import tqdm


class KnowledgeGraph:
    """
    Unified knowledge graph for molecule-protein interaction prediction.

    Stores entities and relations with integer IDs.
    Supports multiple relation types including the pseudo_interaction relation.
    """

    def __init__(self):
        self.entity2id: Dict[str, int] = {}
        self.id2entity: Dict[int, str] = {}
        self.relation2id: Dict[str, int] = {}
        self.id2relation: Dict[int, str] = {}

        self.triples: List[Tuple[int, int, int]] = []  # (head, relation, tail)

        # Entity type tracking
        self.entity_type: Dict[int, str] = {}  # entity_id → 'molecule'/'protein'/'gene'/etc.

        # Molecule and protein sets for MPI tasks
        self.molecule_ids: List[int] = []
        self.protein_ids: List[int] = []

    def add_entity(self, name: str, entity_type: str = 'unknown') -> int:
        """Add entity, return its ID."""
        if name not in self.entity2id:
            idx = len(self.entity2id)
            self.entity2id[name] = idx
            self.id2entity[idx] = name
            self.entity_type[idx] = entity_type

            if entity_type == 'molecule':
                self.molecule_ids.append(idx)
            elif entity_type == 'protein':
                self.protein_ids.append(idx)

        return self.entity2id[name]

    def add_relation(self, name: str) -> int:
        """Add relation type, return its ID."""
        if name not in self.relation2id:
            idx = len(self.relation2id)
            self.relation2id[name] = idx
            self.id2relation[idx] = name
        return self.relation2id[name]

    def add_triple(self, head: str, relation: str, tail: str,
                   head_type: str = 'unknown', tail_type: str = 'unknown'):
        """Add a (head, relation, tail) triple to the KG."""
        h_id = self.add_entity(head, head_type)
        r_id = self.add_relation(relation)
        t_id = self.add_entity(tail, tail_type)
        self.triples.append((h_id, r_id, t_id))

    def get_triples_tensor(self) -> torch.Tensor:
        """Return all triples as (N, 3) tensor."""
        return torch.tensor(self.triples, dtype=torch.long)

    @property
    def num_entities(self) -> int:
        return len(self.entity2id)

    @property
    def num_relations(self) -> int:
        return len(self.relation2id)

    @property
    def num_triples(self) -> int:
        return len(self.triples)

    def summary(self):
        """Print KG statistics."""
        print(f"Knowledge Graph Summary:")
        print(f"  Entities: {self.num_entities}")
        print(f"  Relations: {self.num_relations}")
        print(f"  Triples: {self.num_triples}")
        print(f"  Molecules: {len(self.molecule_ids)}")
        print(f"  Proteins: {len(self.protein_ids)}")
        print(f"  Relation types: {list(self.relation2id.keys())}")

    def save(self, path: str):
        """Save KG to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save({
            'entity2id': self.entity2id,
            'id2entity': self.id2entity,
            'relation2id': self.relation2id,
            'id2relation': self.id2relation,
            'triples': self.triples,
            'entity_type': self.entity_type,
            'molecule_ids': self.molecule_ids,
            'protein_ids': self.protein_ids,
        }, path / 'kg_data.pt')
        print(f"KG saved to {path / 'kg_data.pt'}")

    @classmethod
    def load(cls, path: str) -> 'KnowledgeGraph':
        """Load KG from disk."""
        data = torch.load(Path(path) / 'kg_data.pt')
        kg = cls()
        kg.entity2id = data['entity2id']
        kg.id2entity = data['id2entity']
        kg.relation2id = data['relation2id']
        kg.id2relation = data['id2relation']
        kg.triples = data['triples']
        kg.entity_type = data['entity_type']
        kg.molecule_ids = data['molecule_ids']
        kg.protein_ids = data['protein_ids']
        return kg


def load_chebi_uniprot(data_dir: str, kg: KnowledgeGraph) -> int:
    """
    Load CHEBI-UniProt catalytic activity relations.
    ~1.35M potential molecule-protein catalytic relationships.

    Data format: TSV with columns [chebi_id, uniprot_id, relation_type]
    Download: https://www.ebi.ac.uk/chebi/ + https://www.uniprot.org/
    """
    filepath = Path(data_dir) / 'chebi_uniprot_interactions.tsv'

    if not filepath.exists():
        print(f"  [CHEBI] File not found: {filepath}")
        print(f"  Download from: https://www.ebi.ac.uk/chebi/")
        print(f"  Format: chebi_id\\tuniprot_id\\trelation")
        return 0

    count = 0
    df = pd.read_csv(filepath, sep='\t', header=0)

    for _, row in tqdm(df.iterrows(), total=len(df), desc="  CHEBI-UniProt"):
        kg.add_triple(
            head=str(row.iloc[0]),
            relation='catalytic_activity',
            tail=str(row.iloc[1]),
            head_type='molecule',
            tail_type='protein',
        )
        count += 1

    return count


def load_kegg(data_dir: str, kg: KnowledgeGraph) -> int:
    """
    Load KEGG pathway and gene regulation data.

    Data: gene-pathway, compound-enzyme, compound-pathway relations.
    Download: https://www.kegg.jp/ (requires academic license)
    """
    filepath = Path(data_dir) / 'kegg_relations.tsv'

    if not filepath.exists():
        print(f"  [KEGG] File not found: {filepath}")
        print(f"  Download from: https://www.kegg.jp/kegg/rest/keggapi.html")
        return 0

    count = 0
    df = pd.read_csv(filepath, sep='\t', header=0)

    for _, row in tqdm(df.iterrows(), total=len(df), desc="  KEGG"):
        kg.add_triple(
            head=str(row['head']),
            relation=str(row['relation']),
            tail=str(row['tail']),
            head_type=str(row.get('head_type', 'gene')),
            tail_type=str(row.get('tail_type', 'pathway')),
        )
        count += 1

    return count


def load_gene_ontology(data_dir: str, kg: KnowledgeGraph) -> int:
    """
    Load Gene Ontology annotations.

    Provides: biological_process, cellular_component, molecular_function relations.
    Download: http://geneontology.org/docs/download-go-annotations/
    """
    filepath = Path(data_dir) / 'go_annotations.tsv'

    if not filepath.exists():
        print(f"  [GO] File not found: {filepath}")
        print(f"  Download from: http://geneontology.org/")
        return 0

    count = 0
    df = pd.read_csv(filepath, sep='\t', header=0)

    for _, row in tqdm(df.iterrows(), total=len(df), desc="  Gene Ontology"):
        kg.add_triple(
            head=str(row['protein_id']),
            relation=f"go_{row['aspect']}",  # P/C/F → biological_process/cellular_component/molecular_function
            tail=str(row['go_id']),
            head_type='protein',
            tail_type='go_term',
        )
        count += 1

    return count


def load_pfam(data_dir: str, kg: KnowledgeGraph) -> int:
    """
    Load PFam protein family classifications.

    Download: https://www.ebi.ac.uk/interpro/download/pfam/
    """
    filepath = Path(data_dir) / 'pfam_families.tsv'

    if not filepath.exists():
        print(f"  [PFam] File not found: {filepath}")
        print(f"  Download from: https://www.ebi.ac.uk/interpro/")
        return 0

    count = 0
    df = pd.read_csv(filepath, sep='\t', header=0)

    for _, row in tqdm(df.iterrows(), total=len(df), desc="  PFam"):
        kg.add_triple(
            head=str(row['protein_id']),
            relation='belongs_to_family',
            tail=str(row['pfam_id']),
            head_type='protein',
            tail_type='protein_family',
        )
        count += 1

    return count


def load_enzyme(data_dir: str, kg: KnowledgeGraph) -> int:
    """
    Load ENZYME commission numbers and enzyme-substrate relations.

    Download: https://enzyme.expasy.org/
    """
    filepath = Path(data_dir) / 'enzyme_substrate.tsv'

    if not filepath.exists():
        print(f"  [ENZYME] File not found: {filepath}")
        print(f"  Download from: https://enzyme.expasy.org/")
        return 0

    count = 0
    df = pd.read_csv(filepath, sep='\t', header=0)

    for _, row in tqdm(df.iterrows(), total=len(df), desc="  ENZYME"):
        kg.add_triple(
            head=str(row['substrate']),
            relation='substrate_of',
            tail=str(row['enzyme']),
            head_type='molecule',
            tail_type='protein',
        )
        count += 1

    return count


def load_drugbank(data_dir: str, kg: KnowledgeGraph) -> int:
    """
    Load DrugBank drug-target interactions.

    Download: https://go.drugbank.com/releases/latest (requires academic license)
    """
    filepath = Path(data_dir) / 'drugbank_targets.tsv'

    if not filepath.exists():
        print(f"  [DrugBank] File not found: {filepath}")
        print(f"  Download from: https://go.drugbank.com/")
        return 0

    count = 0
    df = pd.read_csv(filepath, sep='\t', header=0)

    for _, row in tqdm(df.iterrows(), total=len(df), desc="  DrugBank"):
        kg.add_triple(
            head=str(row['drug_id']),
            relation='targets',
            tail=str(row['target_id']),
            head_type='molecule',
            tail_type='protein',
        )
        count += 1

    return count


def load_primekg(data_dir: str, kg: KnowledgeGraph) -> int:
    """
    Alternative: Load from PrimeKG (comprehensive biomedical KG).
    PrimeKG integrates 20+ data sources.

    Download: https://zitniklab.hms.harvard.edu/projects/PrimeKG/
    File: kg.csv with columns [x_id, x_type, relation, y_id, y_type, ...]
    """
    filepath = Path(data_dir) / 'primekg.csv'

    if not filepath.exists():
        # Try downloading
        print(f"  [PrimeKG] Attempting download...")
        try:
            import urllib.request
            url = "https://dataverse.harvard.edu/api/access/datafile/6180620"
            urllib.request.urlretrieve(url, str(filepath))
            print(f"  Downloaded PrimeKG to {filepath}")
        except Exception as e:
            print(f"  [PrimeKG] Download failed: {e}")
            print(f"  Manual download: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/IXA7BM")
            return 0

    count = 0
    # PrimeKG columns: x_index, x_id, x_type, x_name, x_source, relation, display_relation,
    #                   y_index, y_id, y_type, y_name, y_source

    print(f"  Loading PrimeKG from {filepath}...")
    chunks = pd.read_csv(filepath, chunksize=100000)

    for chunk in chunks:
        for _, row in chunk.iterrows():
            x_type = str(row.get('x_type', 'unknown')).lower()
            y_type = str(row.get('y_type', 'unknown')).lower()

            # Map PrimeKG types to our types
            type_map = {
                'drug': 'molecule', 'disease': 'disease', 'gene/protein': 'protein',
                'biological_process': 'go_term', 'cellular_component': 'go_term',
                'molecular_function': 'go_term', 'pathway': 'pathway',
                'anatomy': 'anatomy', 'effect/phenotype': 'phenotype',
            }

            h_type = type_map.get(x_type, x_type)
            t_type = type_map.get(y_type, y_type)

            kg.add_triple(
                head=str(row.get('x_id', row.get('x_index', ''))),
                relation=str(row.get('relation', 'unknown')),
                tail=str(row.get('y_id', row.get('y_index', ''))),
                head_type=h_type,
                tail_type=t_type,
            )
            count += 1

    return count


def build_kg_from_sources(data_dir: str, use_primekg: bool = True) -> KnowledgeGraph:
    """
    Build the full KG from all data sources.

    Args:
        data_dir: directory containing all data files
        use_primekg: if True, use PrimeKG as a single comprehensive source

    Returns:
        Constructed KnowledgeGraph
    """
    kg = KnowledgeGraph()

    # Add pseudo_interaction relation (to be used later)
    kg.add_relation('pseudo_interaction')

    print("Building Knowledge Graph from biological datasets...")

    if use_primekg:
        # PrimeKG is comprehensive and easier to download as a single file
        count = load_primekg(data_dir, kg)
        print(f"  PrimeKG: {count} triples loaded")
    else:
        # Load from individual sources
        sources = [
            ("CHEBI-UniProt", load_chebi_uniprot),
            ("KEGG", load_kegg),
            ("Gene Ontology", load_gene_ontology),
            ("PFam", load_pfam),
            ("ENZYME", load_enzyme),
            ("DrugBank", load_drugbank),
        ]

        for name, loader in sources:
            count = loader(data_dir, kg)
            print(f"  {name}: {count} triples loaded")

    kg.summary()
    return kg
