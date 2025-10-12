import torch
import pandas as pd
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from train_model import ProteinDataset, ProteinLSTM # We reuse the classes from our training script

# --- SETTINGS ---
TEST_DATA_PATH = "test_data.csv"
SAVED_MODEL_PATH = "best_location_model.pth"
BATCH_SIZE = 32

# --- MODEL HYPERPARAMETERS (must match the trained model) ---
VOCAB_SIZE = 22
EMBEDDING_DIM = 128
HIDDEN_DIM = 256

# --- EVALUATION LOGIC ---
# Load test data
test_dataset = ProteinDataset(TEST_DATA_PATH)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# Determine number of classes and location mapping
master_df = pd.read_csv("master_protein_locations.csv")
master_df['location_id'] = master_df['location'].astype('category').cat.codes
num_classes = len(master_df['location_id'].unique())
location_mapping = dict(enumerate(master_df['location'].astype('category').cat.categories))
target_names = [location_mapping[i] for i in range(num_classes)]


# Initialize the model structure
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ProteinLSTM(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, num_classes).to(device)

# Load the saved weights
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

# Print the performance report
print("\n--- Classification Report ---")
print(classification_report(all_labels, all_preds, target_names=target_names))