import torch
from models import prepare_model
import pandas as pd
from utils import load_tokenizer
from dataloader import TestDataset
from torch.utils.data import DataLoader
import numpy as np


def run_stage1_prediction(protein_sequence: str) -> bool:
    """
    Takes a raw protein sequence, creates windows, and predicts ubiquitination.
    Returns True if any ubiquitination site is predicted, False otherwise.
    """
    # --- 1. Hardcode the settings from the original config file ---
    window_size = 55
    checkpoint_path = 'pretrained_models/end_to_end/lstm/55/best_valid_f1_checkpoint.pth'
    device = torch.device('cpu')  # Run this part on CPU for simplicity
    configs = {
        'backbone': 'lstm',
        'window_size': 55
    }

    # --- 2. Create windowed dataframes from the single sequence ---
    all_windows = []
    padding_size = window_size // 2
    padded_sequence = ('-' * padding_size) + protein_sequence + ('-' * padding_size)

    for i in range(len(protein_sequence)):
        window = padded_sequence[i: i + window_size]
        all_windows.append({"window": window})

    if not all_windows:
        print("Warning: Input sequence is empty.")
        return False

    test_df = pd.DataFrame(all_windows)

    # --- 3. Load tokenizer and prepare data for the model ---
    tokenizer = load_tokenizer('.')
    test_data = TestDataset(test_df, tokenizer)
    test_dataloader = DataLoader(test_data, batch_size=128, shuffle=False)

    # --- 4. Load the pre-trained model ---
    model = prepare_model(device, configs, tokenizer, print_params=False)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # --- 5. Run the prediction ---
    results = []
    with torch.no_grad():
        for sequence_batch in test_dataloader:
            sequence_batch = sequence_batch.to(device)
            pred = model(sequence_batch)
            results.append(torch.argmax(torch.softmax(pred.cpu(), dim=-1), dim=-1).numpy())

    final_results = np.concatenate(results)

    # --- 6. Return True if any site was predicted as ubiquitinated (a '1') ---
    if np.any(final_results == 1):
        return True
    else:
        return False