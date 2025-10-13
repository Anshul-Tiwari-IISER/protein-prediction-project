import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import ast
import os

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# --- 1. SETTINGS & HYPERPARAMETERS ---
TRAIN_DATA_PATH = os.path.join(script_dir, "train_data.csv")
VALID_DATA_PATH = os.path.join(script_dir, "validation_data.csv")
MODEL_SAVE_PATH = os.path.join(script_dir, "best_location_model.pth")

# Model Hyperparameters
VOCAB_SIZE = 23
EMBEDDING_DIM = 128
HIDDEN_DIM = 256  # Each direction of the BiLSTM will have this dimension
NUM_EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 0.0005 # A slightly lower learning rate for a more complex model

# --- 2. CUSTOM PYTORCH DATASET ---
class ProteinDataset(Dataset):
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.sequences = self.df['sequence_encoded'].apply(ast.literal_eval).tolist()
        self.labels = self.df['location_id'].tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sequence = torch.tensor(self.sequences[idx], dtype=torch.long)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return sequence, label

# --- 3. UPGRADED MODEL: BiLSTM with Attention ---
class BiLSTM_Attention(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes):
        super(BiLSTM_Attention, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # Bidirectional LSTM
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, num_layers=2, dropout=0.5, bidirectional=True)

        # Attention layer
        self.attention = nn.Linear(hidden_dim * 2, 1) # hidden_dim * 2 because it's bidirectional

        # Final classifier
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)

        # LSTM output will be (batch_size, seq_len, hidden_dim * 2)
        lstm_out, _ = self.lstm(embedded)

        # Attention mechanism
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)

        # Final classification
        out = self.fc(context_vector)

        # --- THIS IS THE ONLY CHANGE ---
        # Return both the prediction and the attention weights
        return out, attention_weights

# --- 4. TRAINING LOGIC ---
if __name__ == '__main__':
    train_dataset = ProteinDataset(TRAIN_DATA_PATH)
    val_dataset = ProteinDataset(VALID_DATA_PATH)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    num_classes = len(pd.read_csv(TRAIN_DATA_PATH)['location_id'].unique())
    print(f"Detected {num_classes} unique locations (classes).")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Use the new model
    model = BiLSTM_Attention(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_accuracy = 0.0

    print("\nStarting model training with BiLSTM + Attention...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_train_loss = 0
        for sequences, labels in train_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        model.eval()
        total_correct = 0
        total_samples = 0
        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences, labels = sequences.to(device), labels.to(device)
                outputs = model(sequences)
                _, predicted = torch.max(outputs.data, 1)
                total_samples += labels.size(0)
                total_correct += (predicted == labels).sum().item()

        val_accuracy = total_correct / total_samples
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {avg_train_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"New best model saved with accuracy: {best_val_accuracy:.4f}")

    print("\nTraining complete!")
    print(f"Best model saved to {MODEL_SAVE_PATH} with accuracy {best_val_accuracy:.4f}")