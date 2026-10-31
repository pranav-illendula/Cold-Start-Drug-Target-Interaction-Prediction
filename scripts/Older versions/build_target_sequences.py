
import pandas as pd

zip_path = 'data/BindingDB_All_202609_tsv.zip'  # <-- adjust to your actual filename/path

# read just the header row to check columns (pandas handles the zip transparently)
header_df = pd.read_csv(zip_path, sep='\t', compression='zip', nrows=0)
columns = header_df.columns.tolist()
print(f"Total columns: {len(columns)}")
for c in columns:
    if 'UniProt' in c:
        print(repr(c))

name_col = 'UniProt (SwissProt) Recommended Name of Target Chain 1'  # confirm/adjust from output above
id_col = 'UniProt (SwissProt) Primary ID of Target Chain 1'          # confirm/adjust from output above

target_to_uniprot = {}
chunk_size = 200_000

for chunk in pd.read_csv(
    zip_path,
    sep='\t',
    compression='zip',
    usecols=[name_col, id_col],
    chunksize=chunk_size,
    low_memory=False,
    on_bad_lines='skip'
):
    chunk = chunk.dropna(subset=[name_col, id_col])
    for name, uid in zip(chunk[name_col], chunk[id_col]):
        if name not in target_to_uniprot:
            target_to_uniprot[name] = uid

print(f"Mapped {len(target_to_uniprot)} unique target names to UniProt IDs")

import pickle
with open('data/splits/target_to_uniprot.pkl', 'wb') as f:
    pickle.dump(target_to_uniprot, f)

train_df = pd.read_csv("data/splits/split_train.csv")
cold_drug_df = pd.read_csv("data/splits/split_cold_drug.csv")
cold_target_df = pd.read_csv("data/splits/split_cold_target.csv")
cold_pair_df = pd.read_csv("data/splits/split_cold_pair.csv")

for df in [train_df, cold_drug_df, cold_target_df, cold_pair_df]:
    df['UniProt_ID'] = df['Target_Name'].map(target_to_uniprot)

for name, df in [('train', train_df), ('cold_drug', cold_drug_df),
                  ('cold_target', cold_target_df), ('cold_pair', cold_pair_df)]:
    matched = df['UniProt_ID'].notna().sum()
    print(f"{name}: {matched}/{len(df)} rows matched ({100*matched/len(df):.1f}%)")

unmatched = train_df[train_df['UniProt_ID'].isna()]['Target_Name'].value_counts()
print(f"Unmatched unique target names: {len(unmatched)}")
print(unmatched.head(15))

train_df = train_df.dropna(subset=['UniProt_ID']).reset_index(drop=True)
cold_drug_df = cold_drug_df.dropna(subset=['UniProt_ID']).reset_index(drop=True)
cold_target_df = cold_target_df.dropna(subset=['UniProt_ID']).reset_index(drop=True)
cold_pair_df = cold_pair_df.dropna(subset=['UniProt_ID']).reset_index(drop=True)

print(train_df.shape, cold_drug_df.shape, cold_target_df.shape, cold_pair_df.shape)

import requests
import time
from tqdm import tqdm

def fetch_uniprot_sequence(uniprot_id):
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            return None
        lines = resp.text.strip().split('\n')
        sequence = ''.join(lines[1:])  # skip the ">header" line
        return sequence
    except Exception:
        return None

# collect all unique UniProt IDs actually used across your splits
all_ids = pd.concat([
    train_df['UniProt_ID'], cold_drug_df['UniProt_ID'],
    cold_target_df['UniProt_ID'], cold_pair_df['UniProt_ID']
]).unique()

print(f"Fetching sequences for {len(all_ids)} unique UniProt IDs...")

sequence_cache = {}
failed_ids = []

for uid in tqdm(all_ids):
    seq = fetch_uniprot_sequence(uid)
    if seq is None:
        failed_ids.append(uid)
    else:
        sequence_cache[uid] = seq
    time.sleep(0.05)  # small delay to be polite to UniProt's servers

print(f"Fetched {len(sequence_cache)} sequences, {len(failed_ids)} failed")

import pickle
with open('data/splits/uniprot_sequence_cache.pkl', 'wb') as f:
    pickle.dump(sequence_cache, f)
print("Saved sequence cache to Drive")

# !pip install fair-esm -q

import torch
import esm

# load a small-but-effective ESM-2 model
model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
batch_converter = alphabet.get_batch_converter()
model.eval()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

print(f"Model loaded on {device}")
print(f"Embedding dimension: {model.embed_dim}")


import pandas as pd
import pickle


with open(f'data/splits/target_to_uniprot.pkl', 'rb') as f:
    target_to_uniprot = pickle.load(f)

with open(f'data/splits/uniprot_sequence_cache.pkl', 'rb') as f:
    sequence_cache = pickle.load(f)

print(f"Cached sequences: {len(sequence_cache)}")

from tqdm import tqdm
import torch
MAX_LEN = 1022  # ESM-2 has a max context; truncate anything longer
BATCH_SIZE = 8   # small batch since sequences can be long; increase if you have headroom

def get_embedding_batch(id_seq_pairs):
    data = [(uid, seq[:MAX_LEN]) for uid, seq in id_seq_pairs]
    labels, strs, tokens = batch_converter(data)
    tokens = tokens.to(device)

    with torch.no_grad():
        results = model(tokens, repr_layers=[12], return_contacts=False)
    token_reps = results["representations"][12]  # layer 12 = final layer for this model size

    embeddings = {}
    for i, (uid, seq) in enumerate(data):
        # mean-pool over actual residues only (skip padding + special tokens)
        seq_len = len(seq)
        rep = token_reps[i, 1:seq_len+1].mean(0)  # skip position 0 (BOS token)
        embeddings[uid] = rep.cpu()
    return embeddings

# sort by length so batches have similar-length sequences (minimizes padding waste)
items = sorted(sequence_cache.items(), key=lambda x: len(x[1]))

target_embeddings = {}
for i in tqdm(range(0, len(items), BATCH_SIZE)):
    batch = items[i:i+BATCH_SIZE]
    embs = get_embedding_batch(batch)
    target_embeddings.update(embs)

print(f"Generated embeddings for {len(target_embeddings)} targets")
print(f"Embedding shape example: {list(target_embeddings.values())[0].shape}")

import pickle

save_dir = 'data/splits'
with open(f'{save_dir}/target_embeddings.pkl', 'wb') as f:
    pickle.dump(target_embeddings, f)

print("Saved target embeddings to Drive")

# verify immediately by reloading
with open(f'{save_dir}/target_embeddings.pkl', 'rb') as f:
    check = pickle.load(f)
print(f"Reloaded {len(check)} embeddings, sample shape: {list(check.values())[0].shape}")


