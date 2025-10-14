import torch
import torch.nn as nn
import numpy as np


# This is the same model architecture as before. No changes are needed here.
class BiLSTM_Attention(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes):
        super(BiLSTM_Attention, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)

        # Attention mechanism
        attention_weights = torch.softmax(self.attention(lstm_out).squeeze(-1), dim=1)
        context_vector = torch.bmm(attention_weights.unsqueeze(1), lstm_out).squeeze(1)

        output = self.fc(context_vector)
        return output, attention_weights


def create_marked_sequence(sequence: str, site_predictions: list):
    """
    Replaces predicted ubiquitination sites in a sequence with a special character 'U'.

    Args:
        sequence (str): The original protein sequence.
        site_predictions (list): A list of tuples, e.g., [(pos, residue, score), ...].

    Returns:
        str: The sequence with ubiquitinated lysines replaced by 'U'.
    """
    # Convert sequence to a list of characters for easy modification
    seq_list = list(sequence)

    # Get just the positions from the prediction list
    predicted_positions = [site[0] for site in site_predictions]

    for pos in predicted_positions:
        # Check if the position is valid and if the residue is indeed a Lysine
        # Position is 1-based, index is 0-based
        if 1 <= pos <= len(seq_list) and seq_list[pos - 1] == 'K':
            seq_list[pos - 1] = 'U'  # 'U' for Ubiquitinated

    return "".join(seq_list)


def preprocess_marked_sequence(sequence: str, vocab: dict, max_len: int):
    """
    Converts a marked sequence into a numerical tensor for the model.
    """
    encoded = [vocab.get(char, 0) for char in sequence]
    if len(encoded) > max_len:
        encoded = encoded[:max_len]
    else:
        # Pad the sequence if it's shorter than max_len
        encoded = encoded + [vocab['<pad>']] * (max_len - len(encoded))
    return torch.tensor(encoded, dtype=torch.long).unsqueeze(0)


# --- Example Usage ---
if __name__ == "__main__":
    # 1. Define our inputs
    test_sequence = "MIVFWAKSKVTSLEEAKDPHYPFKPWKVRFSLFEFNYGPYN"
    # This is the kind of output our MOCK Stage 1 provides
    mock_site_predictions = [(7, 'K', 0.91), (23, 'K', 0.75)]

    print("Original Sequence:", test_sequence)
    print("Predicted Sites:", mock_site_predictions)

    # 2. Create the "marked" sequence
    marked_sequence = create_marked_sequence(test_sequence, mock_site_predictions)
    print("Marked Sequence:  ", marked_sequence)
    print("--- Notice the 'K' at positions 7 and 23 are now 'U' ---")

    # 3. Preprocess the marked sequence for the model
    # Note: We've added our new special character 'U' to the vocabulary!
    amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y',
                   'U', '-']
    vocab = {char: i + 1 for i, char in enumerate(amino_acids)}
    vocab['<pad>'] = 0

    max_sequence_length = 1000
    numerical_input = preprocess_marked_sequence(marked_sequence, vocab, max_sequence_length)

    print("\nVocabulary size:", len(vocab))
    print("Numerical Tensor Shape:", numerical_input.shape)

    # 4. (Placeholder) This tensor would now be ready to be fed into a trained model
    print("\nThis numerical tensor is the final input for the new Stage 2 model.")
    # In the future, we would load a trained model and call: model(numerical_input)