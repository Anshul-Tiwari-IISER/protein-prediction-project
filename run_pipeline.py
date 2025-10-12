# run_pipeline.py

# --- Import necessary functions and classes ---
# You'll need to adjust these paths
from demo.inference_end_to_end import run_stage1_prediction  # We will need to modify this script slightly
from stage2_localization.train_model import ProteinLSTM, ProteinDataset  # Reuse our model definition
import torch

# --- 1. DEFINE YOUR INPUT PROTEIN SEQUENCE ---
my_protein_sequence = "MTEITAAMVKELRESTGAGMMDCKNALSETNGDFDKAVQLLREKGLGKAAKKADRLAAEG"


def main():
    # --- STAGE 1: PREDICT UBIQUITINATION ---
    print("--- Running Stage 1: Ubiquitination Prediction ---")
    # We need to adapt the Stage 1 script to be callable and return a result
    # For now, let's assume it returns True if ubiquitinated
    is_ubiquitinated = run_stage1_prediction(my_protein_sequence)  # This function needs to be created

    if not is_ubiquitinated:
        print("\nResult: The protein is NOT predicted to be ubiquitinated.")
        return  # Stop the pipeline

    print("\nResult: The protein IS predicted to be ubiquitinated.")

    # --- STAGE 2: PREDICT LOCALIZATION ---
    print("\n--- Running Stage 2: Subcellular Localization Prediction ---")

    # Load the trained Stage 2 model
    model = ProteinLSTM(...)  # Initialize with the correct parameters
    model.load_state_dict(torch.load('stage2_localization/best_location_model.pth', map_location=torch.device('cpu')))
    model.eval()

    # Preprocess the input sequence into the numerical format the model expects
    # (This logic would be copied from your preprocess_data.py script)
    numerical_sequence = preprocess_sequence_for_stage2(my_protein_sequence)

    # Make the prediction
    with torch.no_grad():
        output = model(numerical_sequence)
        _, predicted_id = torch.max(output.data, 1)

    # Convert the predicted ID back to a location name (e.g., 0 -> 'nucleus')
    location_name = convert_id_to_location(predicted_id.item())  # This function needs to be created

    print(f"\nFINAL PREDICTION: The protein is likely located in the {location_name}.")


if __name__ == '__main__':
    main()