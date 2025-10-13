import torch
import pandas as pd
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from train_model import ProteinDataset, BiLSTM_Attention # <-- This line is updated
import os

script_dir = os.path.dirname(os.path.abspath(__file__))

# --- SETTINGS ---
TEST_DATA_PATH = os.path.join(script_dir, "test_data.csv")
SAVED_MODEL_PATH = os.path.join(script_dir, "best_location_model.pth")
MASTER_DATA_PATH = os.path.join(script_dir, "master_protein_locations.csv")
BATCH_SIZE = 32

# --- MODEL HYPERPARAMETERS (must match the trained model) ---
VOCAB_SIZE = 23
EMBEDDING_DIM = 128
HIDDEN_DIM = 256

# --- EVALUATION LOGIC ---
test_dataset = ProteinDataset(TEST_DATA_PATH)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

master_df = pd.read_csv(MASTER_DATA_PATH)
master_df['location_id'] = master_df['location'].astype('category').cat.codes
num_classes = len(master_df['location_id'].unique())
location_mapping = dict(enumerate(master_df['location'].astype('category').cat.categories))
target_names = [location_mapping[i] for i in range(num_classes)]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Use the new model definition
model = BiLSTM_Attention(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, num_classes).to(device) # <-- This line is updated

model.load_state_dict(torch.load(SAVED_MODEL_PATH))
model.eval()

print(f"Model loaded from {SAVED_MODEL_PATH}. Evaluating on test data...")

all_preds = []
all_labels = []
with torch.no_grad():
    for sequences, labels in test_loader:
        sequences, labels = sequences.to(device), labels.to(device)
        outputs = model(sequences)
        _, predicted = torch.max(outputs.data, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print("\n--- Classification Report ---")
print(classification_report(all_labels, all_preds, target_names=target_names))