import torch
import torch.nn as nn
import numpy as np
import sys
import os
import argparse # <-- NEW: Import the argument parser

# Add the stage2_localization subfolder to the path
sys.path.append(os.path.abspath('stage2_localization'))

from inference_end_to_end import run_stage1_prediction
from train_model import BiLSTM_Attention

# --- NEW: Setup the command-line argument parser ---
parser = argparse.ArgumentParser(description="Run the full protein prediction pipeline.")
parser.add_argument("sequence", type=str, help="The protein amino acid sequence to analyze.")

# --- 2. DEFINE HELPER FUNCTIONS FOR STAGE 2 ---
def preprocess_for_stage2(sequence: str, vocab: dict, max_len: int):
    encoded = [vocab.get(char, 0) for char in sequence]
    if len(encoded) > max_len:
        encoded = encoded[:max_len]
    else:
        encoded = encoded + [vocab['<pad>']] * (max_len - len(encoded))
    return torch.tensor(encoded, dtype=torch.long).unsqueeze(0)

# --- 3. MAIN PIPELINE LOGIC ---
def main(protein_sequence): # <-- UPDATED: Accepts the sequence as an argument
    print("--- Starting Integrated Protein Prediction Pipeline ---")
    print(f"Input Sequence: {protein_sequence[:50]}...")

    # --- STAGE 1: PREDICT UBIQUITINATION ---
    print("\n--- Running Stage 1: Ubiquitination Prediction ---")
    is_ubiquitinated = run_stage1_prediction(protein_sequence)

    if not is_ubiquitinated:
        print("\nPIPELINE RESULT: The protein is NOT predicted to be ubiquitinated.")
        return

    print("\nResult: The protein IS predicted to be ubiquitinated.")

    # --- STAGE 2: PREDICT LOCALIZATION ---
    print("\n--- Running Stage 2: Subcellular Localization Prediction ---")

    device = torch.device('cpu')
    model_path = 'stage2_localization/best_location_model.pth'

    # Model settings must match those used during training
    vocab_size = 23
    embedding_dim = 128
    hidden_dim = 256
    num_classes = 5
    max_sequence_length = 1000

    location_mapping = {0: 'Endoplasmic', 1: 'Golgi', 2: 'cytoplasm', 3: 'mitochondrion', 4: 'nucleus'}

    amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', '-', 'X', 'B', 'Z', 'J', 'U', 'O']
    vocab = {char: i + 1 for i, char in enumerate(amino_acids)}
    vocab['<pad>'] = 0

    model = BiLSTM_Attention(vocab_size, embedding_dim, hidden_dim, num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    numerical_sequence = preprocess_for_stage2(protein_sequence, vocab, max_sequence_length).to(device)

    with torch.no_grad():
        output, _ = model(numerical_sequence) # Updated model returns two values
        _, predicted_id = torch.max(output.data, 1)
        location_name = location_mapping.get(predicted_id.item(), "Unknown")

    print(f"\nPIPELINE RESULT: The protein is likely located in the '{location_name}'.")

if __name__ == '__main__':
    args = parser.parse_args() # <-- NEW: Get the arguments from the command line
    main(args.sequence) # <-- UPDATED: Pass the sequence to the main function