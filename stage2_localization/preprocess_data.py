import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import os

# --- 1. SETTINGS (Now with smarter path handling) ---
# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
# Construct the full path to the input CSV file
input_csv_file = os.path.join(script_dir, "master_protein_locations.csv")
max_sequence_length = 1000 # We will trim or pad sequences to this length

# --- 2. THE SCRIPT LOGIC (No need to change) ---
print(f"Reading data from {input_csv_file}...")
df = pd.read_csv(input_csv_file)

# --- Convert location labels to numbers ---
# Create a mapping from location name to a number (e.g., nucleus -> 0)
df['location_id'] = df['location'].astype('category').cat.codes
location_mapping = dict(enumerate(df['location'].astype('category').cat.categories))
print("\nLocation mapping created:")
print(location_mapping)

# --- Convert protein sequences to numbers (Tokenization) ---
# Create a vocabulary of all possible amino acids
amino_acids = sorted(list(set(''.join(df['sequence']))))
# Add a token for "padding"
vocab = {char: i + 1 for i, char in enumerate(amino_acids)}
vocab['<pad>'] = 0 # Padding token
print(f"\nVocabulary size: {len(vocab)}")

def sequence_to_integers(sequence):
    """Converts a protein sequence string into a list of integers."""
    encoded = [vocab.get(char, 0) for char in sequence]
    # Pad or truncate the sequence to the max_sequence_length
    if len(encoded) > max_sequence_length:
        return encoded[:max_sequence_length] # Truncate
    else:
        return encoded + [vocab['<pad>']] * (max_sequence_length - len(encoded)) # Pad

print("\nConverting sequences to numerical format...")
df['sequence_encoded'] = df['sequence'].apply(sequence_to_integers)

# --- Split the data into training (80%), validation (10%), and test (10%) sets ---
print("\nSplitting data into train, validation, and test sets...")
# First, split into training and a temporary set (validation + test)
train_df, temp_df = train_test_split(
    df,
    test_size=0.2, # 20% for temp set
    random_state=42,
    stratify=df['location_id'] # Ensures each set has a similar mix of locations
)

# Now split the temporary set into validation and test sets
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5, # 50% of the temp set (which is 10% of the original)
    random_state=42,
    stratify=temp_df['location_id']
)

print(f"Training set size: {len(train_df)}")
print(f"Validation set size: {len(val_df)}")
print(f"Test set size: {len(test_df)}")

# --- Save the new datasets ---
# We only need the encoded sequence and the location ID for the model
train_df[['sequence_encoded', 'location_id']].to_csv(os.path.join(script_dir, 'train_data.csv'), index=False)
val_df[['sequence_encoded', 'location_id']].to_csv(os.path.join(script_dir, 'validation_data.csv'), index=False)
test_df[['sequence_encoded', 'location_id']].to_csv(os.path.join(script_dir, 'test_data.csv'), index=False)

print("\nSuccessfully created train_data.csv, validation_data.csv, and test_data.csv in the stage2_localization folder!")