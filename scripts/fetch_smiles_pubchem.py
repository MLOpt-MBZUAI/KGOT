"""Fetch SMILES for all DrugBank drugs via PubChem API (DrugBank ID → CID → SMILES)."""
import urllib.request, json, time, torch, numpy as np, sys
from pathlib import Path
from tqdm import tqdm

def get_cid_from_drugbank(dbid, timeout=8):
    """DrugBank ID → PubChem CID."""
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/substance/sourceid/DrugBank/{dbid}/cids/JSON"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
            cids = data.get("InformationList", {}).get("Information", [{}])[0].get("CID", [])
            return cids[0] if cids else None
    except:
        return None

def get_smiles_from_cid(cid, timeout=8):
    """PubChem CID → Canonical SMILES."""
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/CanonicalSMILES/TXT"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode().strip()
    except:
        return None

def batch_get_smiles(cids, timeout=10):
    """Batch CIDs → SMILES (up to 100 at a time)."""
    cid_str = ",".join(str(c) for c in cids)
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid_str}/property/CanonicalSMILES/JSON"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
            results = {}
            for prop in data.get("PropertyTable", {}).get("Properties", []):
                results[prop["CID"]] = prop["CanonicalSMILES"]
            return results
    except:
        return {}

def main():
    kg_data = torch.load("data/kg/kg_data.pt", weights_only=False)
    id2entity = kg_data["id2entity"]
    mpi_mol_ids = np.load("data/kg/mpi_molecule_ids.npy")

    # Extract DrugBank IDs
    drugbank_ids = [id2entity[int(mid)].replace("DRUG:", "") for mid in mpi_mol_ids]
    print(f"Total drugs to look up: {len(drugbank_ids)}")

    # Step 1: DrugBank ID → CID (batch of 1 at a time due to API limit)
    print("\nStep 1: DrugBank ID → PubChem CID...")
    db_to_cid = {}
    for i, dbid in enumerate(tqdm(drugbank_ids, desc="CID lookup")):
        if dbid in db_to_cid:
            continue
        cid = get_cid_from_drugbank(dbid)
        if cid:
            db_to_cid[dbid] = cid
        # Rate limit: 5 requests/second
        if i % 5 == 4:
            time.sleep(1.0)
        # Save progress every 500
        if i % 500 == 499:
            print(f"  Progress: {i+1}/{len(drugbank_ids)}, found {len(db_to_cid)} CIDs")
            np.save("data/embeddings/db_to_cid_partial.npy", db_to_cid)

    print(f"Got CIDs for {len(db_to_cid)} / {len(set(drugbank_ids))} unique drugs")

    # Step 2: Batch CID → SMILES
    print("\nStep 2: CID → SMILES (batch)...")
    all_cids = list(db_to_cid.values())
    cid_to_smiles = {}

    batch_size = 100
    for i in tqdm(range(0, len(all_cids), batch_size), desc="SMILES batch"):
        batch = all_cids[i:i+batch_size]
        results = batch_get_smiles(batch)
        cid_to_smiles.update(results)
        time.sleep(0.5)

    print(f"Got SMILES for {len(cid_to_smiles)} / {len(all_cids)} CIDs")

    # Build final SMILES list
    smiles_list = []
    found = 0
    for dbid in drugbank_ids:
        cid = db_to_cid.get(dbid)
        if cid and cid in cid_to_smiles:
            smiles_list.append(cid_to_smiles[cid])
            found += 1
        else:
            smiles_list.append(None)

    print(f"\nFinal: {found} / {len(drugbank_ids)} drugs have real SMILES ({100*found/len(drugbank_ids):.1f}%)")

    # Save
    np.save("data/embeddings/drug_smiles_real.npy", np.array(smiles_list, dtype=object))
    print("Saved data/embeddings/drug_smiles_real.npy")
    print(f"Sample: {[(drugbank_ids[i], smiles_list[i][:40] if smiles_list[i] else None) for i in range(10)]}")

if __name__ == "__main__":
    main()
