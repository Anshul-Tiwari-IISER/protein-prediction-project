import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from tqdm import tqdm


# --- Model Architecture (same as in the pipeline script) ---
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


# --- Main Training Script ---
if __name__ == "__main__":
    # 1. Load the new dataset
    data_path = 'stage2_localization/marked_location_data.csv'
    df = pd.read_csv(data_path)

    # 2. Define the vocabulary (including 'U')
    amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y',
                   'U', '-']
    vocab = {char: i + 1 for i, char in enumerate(amino_acids)}
    vocab['<pad>'] = 0

    # 3. Preprocess the sequences
    max_len = 1000
    sequences = df['marked_sequence'].values
    encoded_sequences = [[vocab.get(char, 0) for char in seq] for seq in sequences]
    padded_sequences = [seq + [vocab['<pad>']] * (max_len - len(seq)) if len(seq) < max_len else seq[:max_len] for seq
                        in encoded_sequences]

    X = np.array(padded_sequences)

    # 4. Encode the labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['location'].values)

    # 5. Split the data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 6. Create PyTorch DataLoaders
    train_data = TensorDataset(torch.LongTensor(X_train), torch.LongTensor(y_train))
    val_data = TensorDataset(torch.LongTensor(X_val), torch.LongTensor(y_val))

    batch_size = 16
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    val_loader = DataLoader(val_data, shuffle=False, batch_size=batch_size)

    # 7. Initialize the model, loss, and optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = BiLSTM_Attention(
        vocab_size=len(vocab),
        embedding_dim=128,
        hidden_dim=256,
        num_classes=len(label_encoder.classes_)
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 8. Training Loop
    epochs = 10  # Train for 10 epochs
    best_val_f1 = 0.0

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0

        for sequences, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Training]"):
            sequences, labels = sequences.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs, _ = model(sequences)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        # Validation
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences, labels = sequences.to(device), labels.to(device)
                outputs, _ = model(sequences)
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_f1 = f1_score(all_labels, all_preds, average='weighted')
        avg_train_loss = total_train_loss / len(train_loader)

        print(f"Epoch {epoch + 1}/{epochs} -> Train Loss: {avg_train_loss:.4f}, Validation F1: {val_f1:.4f}")

        # Save the best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), 'best_location_model_v2.pth')
            print(f"  -> New best model saved with F1-score: {best_val_f1:.4f}")

    print("\nTraining complete!")
    print(f"Final best model saved as 'best_location_model_v2.pth' with a validation F1-score of {best_val_f1:.4f}")