import torch
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Add the stage2_localization subfolder to the path
sys.path.append(os.path.abspath('stage2_localization'))

from train_model import BiLSTM_Attention # Import our updated model

# --- 1. SETTINGS AND INPUT ---
# You can change this to any protein sequence you want to analyze
PROTEIN_SEQUENCE = "MKSFFTWPSGPQLERLDTLCLATVTCLALFSLFLGISSAAQANQRITTESLPPKIMELVPNPNNTGLSKNEQSTQTHYSENLSASPYKFENPYFEYLEIN"
MODEL_PATH = 'stage2_localization/best_location_model.pth'

# --- 2. PREPROCESSING LOGIC (Copied from run_pipeline.py) ---
def preprocess_for_stage2(sequence: str, vocab: dict, max_len: int):
    encoded = [vocab.get(char, 0) for char in sequence]
    if len(encoded) > max_len:
        encoded = encoded[:max_len]
    else:
        encoded = encoded + [vocab['<pad>']] * (max_len - len(encoded))
    return torch.tensor(encoded, dtype=torch.long).unsqueeze(0)

# --- 3. MAIN VISUALIZATION LOGIC ---
def main():
    print("--- Running Attention Visualization ---")

    # --- Load Model and Settings ---
    device = torch.device('cpu') # Run on CPU for this task

    # These settings must match your trained model
    vocab_size = 23
    embedding_dim = 128
    hidden_dim = 256
    num_classes = 5
    max_sequence_length = 1000

    # Load the model architecture
    model = BiLSTM_Attention(vocab_size, embedding_dim, hidden_dim, num_classes).to(device)
    # Load the saved weights
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    # --- Preprocess the Input Sequence ---
    amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', '-', 'X', 'B', 'Z', 'J', 'U', 'O']
    vocab = {char: i + 1 for i, char in enumerate(amino_acids)}
    vocab['<pad>'] = 0

    numerical_sequence = preprocess_for_stage2(PROTEIN_SEQUENCE, vocab, max_sequence_length)

    # --- Get Prediction and Attention Weights ---
    with torch.no_grad():
        # Our model now returns two items
        prediction, attention_weights = model(numerical_sequence.to(device))

    # Squeeze to remove batch and extra dimensions
    attention_weights = attention_weights.squeeze().cpu().numpy()

    # Get only the weights for the actual sequence, not the padding
    actual_sequence_length = len(PROTEIN_SEQUENCE)
    attention_weights = attention_weights[:actual_sequence_length]

    # --- Create and Save the Plot ---
    plt.figure(figsize=(20, 2))
    sns.heatmap([attention_weights], cmap="viridis", cbar=True)
    plt.xticks(ticks=np.arange(actual_sequence_length) + 0.5, labels=list(PROTEIN_SEQUENCE), rotation=90, fontsize=8)
    plt.yticks([])
    plt.title('Attention Weights Heatmap for Protein Sequence')

    output_filename = "attention_heatmap.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')

    print(f"\nSuccessfully generated and saved the visualization as '{output_filename}'")

if __name__ == '__main__':
    main()