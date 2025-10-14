import torch
import torch.nn as nn
import sys
import os

# --- Import Functions from Your New Modules ---
sys.path.append(os.path.abspath('.'))
from stage1_site_predictor import get_ubiquitination_sites
from stage2_site_location_predictor import create_marked_sequence, preprocess_marked_sequence


# --- CORRECTED Model Architecture Definition ---
# This class now perfectly matches the architecture of the saved model.
class BiLSTM_Attention(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes):
        super(BiLSTM_Attention, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # The FIX: Added num_layers=2 and dropout=0.5 to match the trained model
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


# --- Main Pipeline Function ---
def run_enhanced_pipeline(sequence: str):
    """Runs the full, enhanced two-stage prediction pipeline."""
    print("--- Starting Enhanced Pipeline ---")

    # === STAGE 1: Predict Ubiquitination Sites ===
    predicted_sites = get_ubiquitination_sites(sequence)

    if not predicted_sites:
        print("\nPipeline Result: Stage 1 predicted NO ubiquitination sites.")
        return

    print("\nStage 1 Result (Mock): Predicted ubiquitination at the following sites:")
    for pos, res, scr in predicted_sites:
        print(f"  - Position: {pos}, Residue: {res}, Score: {scr}")

    # === STAGE 2: Predict Subcellular Location ===
    print("\n--- Preparing for Stage 2 ---")

    marked_sequence = create_marked_sequence(sequence, predicted_sites)
    print("Created Marked Sequence:", marked_sequence)

    device = torch.device('cpu')
    model_path = 'best_location_model_v2.pth'

    # Model parameters must match the saved model
    vocab_size = 23
    embedding_dim = 128
    hidden_dim = 256
    num_classes = 5

    model = BiLSTM_Attention(vocab_size, embedding_dim, hidden_dim, num_classes).to(device)

    # The FIX: Added weights_only=True to silence the warning
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    print("Loaded pre-trained Stage 2 model successfully.")

    amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y',
                   'U', '-']
    vocab = {char: i + 1 for i, char in enumerate(amino_acids)}
    vocab['<pad>'] = 0
    max_sequence_length = 1000

    numerical_input = preprocess_marked_sequence(marked_sequence, vocab, max_sequence_length)

    location_mapping = {0: 'Endoplasmic', 1: 'Golgi', 2: 'cytoplasm', 3: 'mitochondrion', 4: 'nucleus'}
    with torch.no_grad():
        output, _ = model(numerical_input)
        _, predicted_id = torch.max(output.data, 1)
        location_name = location_mapping.get(predicted_id.item(), "Unknown")

    print("\n--- Final Pipeline Result ---")
    print(f"Predicted Subcellular Location: {location_name}")


# --- Example Usage ---
if __name__ == "__main__":
    test_sequence = "MIVFWARSVTSLEEAKDPHYPFKPWKVRFSLFEFNYGPYN" \
                    "GREGTRLWRFRWENGEKINTWEGPEGTFGVVFLEENVFNS" \
                    "VVERLEIKKSKGKQNKLDLSNLVIPGVEGIDISETFEVIF" \
                    "TDREYEPVTLTVFQSFKVRWQNLKHMVVFVRIG"
    run_enhanced_pipeline(test_sequence)