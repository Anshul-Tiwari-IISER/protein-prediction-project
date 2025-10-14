import streamlit as st
import torch
import sys
import os
import numpy as np

# --- 1. SETUP AND MODEL LOADING ---

# Add the stage2_localization subfolder to the path so we can import its files
sys.path.append(os.path.abspath('stage2_localization'))

# Import all the necessary functions and classes from your project
from inference_end_to_end import run_stage1_prediction
from train_model import BiLSTM_Attention

# Define a function to preprocess the sequence for the Stage 2 model
def preprocess_for_stage2(sequence: str, vocab: dict, max_len: int):
    encoded = [vocab.get(char, 0) for char in sequence]
    if len(encoded) > max_len:
        encoded = encoded[:max_len]
    else:
        encoded = encoded + [vocab['<pad>']] * (max_len - len(encoded))
    return torch.tensor(encoded, dtype=torch.long).unsqueeze(0)

# Use a Streamlit cache decorator to load the model only once
@st.cache_resource
def load_model():
    """Loads the trained BiLSTM+Attention model and its settings."""
    device = torch.device('cpu')
    model_path = 'stage2_localization/best_location_model.pth'

    # Model settings must match those used during training
    vocab_size = 23
    embedding_dim = 128
    hidden_dim = 256
    num_classes = 5

    # Load the model architecture and the saved weights
    model = BiLSTM_Attention(vocab_size, embedding_dim, hidden_dim, num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

# --- 2. STREAMLIT USER INTERFACE ---

st.title("🔬 Protein Localization Predictor")

st.markdown("""
This web app runs a two-stage deep learning pipeline to predict the subcellular
location of ubiquitinated proteins.
""")

# --- Input Text Area for Protein Sequence ---
sequence_input = st.text_area("Enter a protein sequence (in FASTA format or raw)", height=250)

# --- Run Prediction Button ---
if st.button("Predict Location"):
    if sequence_input:
        # Clean the input sequence (handles FASTA headers)
        protein_sequence = "".join(sequence_input.splitlines()[1:]) if sequence_input.startswith('>') else "".join(sequence_input.splitlines())
        protein_sequence = protein_sequence.strip().upper()

        st.write("---")
        st.subheader("Stage 1: Ubiquitination Prediction")

        # Run Stage 1
        with st.spinner("Analyzing ubiquitination status..."):
            is_ubiquitinated = run_stage1_prediction(protein_sequence)

        if not is_ubiquitinated:
            st.warning("Result: The protein is NOT predicted to be ubiquitinated.")
            st.stop()

        st.success("Result: The protein IS predicted to be ubiquitinated.")

        st.write("---")
        st.subheader("Stage 2: Subcellular Localization Prediction")

        # Run Stage 2
        with st.spinner("Predicting final location..."):
            model = load_model()

            # Settings needed for preprocessing
            max_sequence_length = 1000
            location_mapping = {0: 'Endoplasmic', 1: 'Golgi', 2: 'cytoplasm', 3: 'mitochondrion', 4: 'nucleus'}
            amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', '-', 'X', 'B', 'Z', 'J', 'U', 'O']
            vocab = {char: i + 1 for i, char in enumerate(amino_acids)}
            vocab['<pad>'] = 0

            numerical_sequence = preprocess_for_stage2(protein_sequence, vocab, max_sequence_length)

            with torch.no_grad():
                output, _ = model(numerical_sequence)
                _, predicted_id = torch.max(output.data, 1)
                location_name = location_mapping.get(predicted_id.item(), "Unknown")

        st.success(f"Final Prediction: The protein is likely located in the **{location_name}**.")

    else:
        st.error("Please enter a protein sequence to get a prediction.")