import os
import pandas as pd
from Bio import SeqIO

# --- 1. SETTINGS ---
data_folder = "."
output_csv_file = "master_protein_locations.csv"

# --- 2. THE SCRIPT LOGIC (Now with balancing) ---
all_proteins_by_location = {}

fasta_files = [f for f in os.listdir(data_folder) if f.endswith('.fasta')]
print(f"Found files: {fasta_files}")

# First, load all proteins into a dictionary grouped by location
for file_name in fasta_files:
    location_label = file_name.split('_')[0]
    file_path = os.path.join(data_folder, file_name)

    sequences = []
    for record in SeqIO.parse(file_path, "fasta"):
        sequences.append(str(record.seq))

    all_proteins_by_location[location_label] = sequences
    print(f"Loaded {len(sequences)} proteins for location '{location_label}'")

# --- Balancing Logic Starts Here ---
# Find the size of the smallest class
min_size = min(len(seqs) for seqs in all_proteins_by_location.values())
print(f"\nBalancing dataset. Smallest class has {min_size} proteins. Undersampling to this size.")

balanced_proteins = []
# Create the new balanced dataset
for location, sequences in all_proteins_by_location.items():
    # Randomly sample 'min_size' proteins from each location
    df_temp = pd.DataFrame({'sequence': sequences})
    sampled_df = df_temp.sample(n=min_size, random_state=42)

    for seq in sampled_df['sequence']:
        balanced_proteins.append({
            "sequence": seq,
            "location": location
        })

# Convert the balanced list into a final DataFrame
df = pd.DataFrame(balanced_proteins)

# Shuffle the final dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

df.to_csv(output_csv_file, index=False)

print(f"\nSuccessfully created BALANCED master dataset at: {output_csv_file}")
print(f"Total proteins in balanced dataset: {len(df)}")
print("\nBalanced dataset preview:")
print(df.head())
print("\nCounts per location in new dataset:")
print(df['location'].value_counts())