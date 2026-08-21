"""
Generate protein embeddings using ESM-2 (via HuggingFace transformers).
Steps:
1. Map NCBI Gene IDs → gene names (from PrimeKG)
2. Get protein sequences from UniProt (via gene name query)
3. Encode with ESM-2 to get 512-dim embeddings
"""
import os, sys, time, json, urllib.request, urllib.parse
import numpy as np, torch, pandas as pd
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

def get_gene_names_from_primekg(mpi_prot_ids, kg_data):
    """Get gene names/symbols from PrimeKG for our protein IDs."""
    id2entity = kg_data["id2entity"]

    # PrimeKG protein entries have NCBI Gene IDs as y_id
    # Load PrimeKG to get gene_id → gene_name mapping
    gene_id_to_name = {}
    chunks = pd.read_csv("data/raw/primekg/kg.csv", chunksize=200000,
                         usecols=["y_id", "y_type", "y_name"])
    for chunk in chunks:
        prots = chunk[chunk["y_type"] == "gene/protein"].drop_duplicates("y_id")
        for _, row in prots.iterrows():
            gene_id_to_name[str(row["y_id"])] = str(row["y_name"])

    # Map our protein entity IDs to gene names
    names = []
    for pid in mpi_prot_ids:
        entity = id2entity[int(pid)]  # e.g. "PROTEIN:2157"
        gene_id = entity.replace("PROTEIN:", "")
        name = gene_id_to_name.get(gene_id, "")
        names.append(name)

    found = sum(1 for n in names if n)
    print(f"Gene names found: {found} / {len(names)}")
    return names


def fetch_sequences_from_uniprot(gene_names, organism="human", max_fetch=3094):
    """Fetch protein sequences from UniProt REST API by gene name."""
    print(f"Fetching sequences from UniProt for {min(len(gene_names), max_fetch)} proteins...")

    sequences = [None] * len(gene_names)
    fetched = 0

    for i, name in enumerate(tqdm(gene_names[:max_fetch], desc="UniProt")):
        if not name or name == "nan":
            continue

        # UniProt REST API: search by gene name + organism
        query = urllib.parse.quote(f"gene_exact:{name} AND organism_id:9606")
        url = f"https://rest.uniprot.org/uniprotkb/search?query={query}&fields=sequence&format=json&size=1"

        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=8) as resp:
                data = json.loads(resp.read())
                results = data.get("results", [])
                if results:
                    seq = results[0].get("sequence", {}).get("value", "")
                    if seq and len(seq) > 10:
                        sequences[i] = seq
                        fetched += 1
        except:
            pass

        # Rate limit
        if i % 10 == 9:
            time.sleep(0.5)

        # Save progress
        if i % 500 == 499:
            print(f"  Progress: {i+1}, fetched {fetched} sequences")
            np.save("data/embeddings/prot_sequences_partial.npy",
                    np.array(sequences, dtype=object))

    print(f"Total sequences fetched: {fetched} / {len(gene_names)}")
    np.save("data/embeddings/prot_sequences.npy", np.array(sequences, dtype=object))
    return sequences


def encode_with_esm2(sequences, output_path, device="cuda:2", batch_size=8, max_len=1022):
    """Encode protein sequences with ESM-2 via HuggingFace."""
    from transformers import AutoTokenizer, AutoModel

    print("Loading ESM-2 model...")
    model_name = "facebook/esm2_t33_650M_UR50D"  # 650M params, good balance
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()

    print(f"Encoding {len(sequences)} proteins with ESM-2...")
    all_embs = []

    for i in tqdm(range(0, len(sequences), batch_size), desc="ESM-2"):
        batch_seqs = []
        for j in range(i, min(i + batch_size, len(sequences))):
            seq = sequences[j]
            if seq and len(seq) > 10:
                # Truncate long sequences
                batch_seqs.append(seq[:max_len])
            else:
                batch_seqs.append("MAAAA")  # minimal placeholder

        inputs = tokenizer(batch_seqs, return_tensors="pt", padding=True,
                          truncation=True, max_length=max_len + 2).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            # Use mean pooling of last hidden state as embedding
            attention_mask = inputs["attention_mask"].unsqueeze(-1)
            embs = (outputs.last_hidden_state * attention_mask).sum(1) / attention_mask.sum(1)
            all_embs.append(embs.cpu())

    embeddings = torch.cat(all_embs, dim=0)
    # Project to 512 dim if needed (ESM-2 650M outputs 1280-dim)
    if embeddings.size(1) != 512:
        proj = torch.randn(embeddings.size(1), 512) / np.sqrt(embeddings.size(1))
        embeddings = embeddings @ proj

    embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
    torch.save(embeddings, output_path)
    print(f"Saved: {embeddings.shape} → {output_path}")
    return embeddings


def main():
    device = "cuda:2" if torch.cuda.is_available() else "cpu"

    # Load data
    kg_data = torch.load("data/kg/kg_data.pt", weights_only=False)
    mpi_prot_ids = np.load("data/kg/mpi_protein_ids.npy")
    print(f"Proteins to encode: {len(mpi_prot_ids)}")

    # Step 1: Get gene names
    print("\n=== Step 1: Get gene names ===")
    gene_names = get_gene_names_from_primekg(mpi_prot_ids, kg_data)

    # Step 2: Fetch sequences from UniProt
    print("\n=== Step 2: Fetch sequences ===")
    seq_path = Path("data/embeddings/prot_sequences.npy")
    if seq_path.exists():
        sequences = list(np.load(str(seq_path), allow_pickle=True))
        fetched = sum(1 for s in sequences if s is not None)
        print(f"Loaded cached sequences: {fetched} / {len(sequences)}")
    else:
        sequences = fetch_sequences_from_uniprot(gene_names)

    # Step 3: Encode with ESM-2
    print("\n=== Step 3: Encode with ESM-2 ===")
    encode_with_esm2(sequences, "data/embeddings/prot_embeddings.pt", device=device)

    print("\nDone!")


if __name__ == "__main__":
    main()
