import torch
import torch.nn as nn
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- Import Functions from Your New Modules ---
sys.path.append(os.path.abspath('.'))
from stage1_site_predictor import get_ubiquitination_sites
from stage2_site_location_predictor import create_marked_sequence, preprocess_marked_sequence


# --- CORRECTED Model Architecture Definition ---
class BiLSTM_Attention(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes):
        super(BiLSTM_Attention, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=2,
                            bidirectional=True, batch_first=True, dropout=0.5)
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        attention_weights = torch.softmax(self.attention(lstm_out).squeeze(-1), dim=1)
        context_vector = torch.bmm(attention_weights.unsqueeze(1), lstm_out).squeeze(1)
        output = self.fc(context_vector)
        return output, attention_weights


def plot_attention_heatmap(sequence, attention_weights, filename="attention_heatmap.png"):
    """Generates and saves a heatmap of the attention scores."""
    attention_scores = attention_weights.squeeze().cpu().numpy()
    seq_list = list(sequence)

    # Truncate sequence if it's longer than the attention scores
    if len(seq_list) > len(attention_scores):
        seq_list = seq_list[:len(attention_scores)]

    fig, ax = plt.subplots(figsize=(20, 2))
    sns.heatmap([attention_scores], xticklabels=seq_list, yticklabels=False, cmap="viridis", cbar=True, ax=ax)
    ax.set_title("Attention Scores Heatmap")
    plt.savefig(filename)
    print(f"\nAttention heatmap saved to {filename}")
    plt.close()


# --- Main Pipeline Function ---
def run_enhanced_pipeline(sequence: str):
    """Runs the full, enhanced two-stage prediction pipeline."""
    print("--- Starting Enhanced Pipeline ---")

    predicted_sites = get_ubiquitination_sites(sequence)

    if not predicted_sites:
        print("\nPipeline Result: Stage 1 predicted NO ubiquitination sites.")
        return

    print("\nStage 1 Result (Mock): Predicted ubiquitination at:")
    print("  " + ", ".join([f"Lysine {site[0]}" for site in predicted_sites]))

    print("\n--- Preparing for Stage 2 ---")

    marked_sequence = create_marked_sequence(sequence, predicted_sites)
    print("Created Marked Sequence for Model:", marked_sequence)

    device = torch.device('cpu')
    model_path = 'best_location_model_v2.pth'  # Using the new model

    amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y',
                   'U', '-']
    vocab = {char: i + 1 for i, char in enumerate(amino_acids)}
    vocab['<pad>'] = 0

    model = BiLSTM_Attention(
        vocab_size=len(vocab),
        embedding_dim=128,
        hidden_dim=256,
        num_classes=5
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    print("Loaded NEW pre-trained Stage 2 model (v2) successfully.")

    max_sequence_length = 1000
    numerical_input = preprocess_marked_sequence(marked_sequence, vocab, max_sequence_length)

    location_mapping = {0: 'Endoplasmic', 1: 'Golgi', 2: 'cytoplasm', 3: 'mitochondrion', 4: 'nucleus'}
    with torch.no_grad():
        output, attention = model(numerical_input)
        _, predicted_id = torch.max(output.data, 1)
        location_name = location_mapping.get(predicted_id.item(), "Unknown")

    print("\n--- Final Pipeline Output ---")
    print(f"Predicted Subcellular Location: {location_name}")

    # Generate and save the heatmap
    plot_attention_heatmap(marked_sequence, attention)


# --- Example Usage ---
if __name__ == "__main__":
    test_sequence = "MIVFWARSVTSLEEAKDPHYPFKPWKVRFSLFEFNYGPYN" \
                    "GREGTRLWRFRWENGEKINTWEGPEGTFGVVFLEENVFNS" \
                    "VVERLEIKKSKGKQNKLDLSNLVIPGVEGIDISETFEVIF" \
                    "TDREYEPVTLTVFQSFKVRWQNLKHMVVFVRIG"
    run_enhanced_pipeline(test_sequence)