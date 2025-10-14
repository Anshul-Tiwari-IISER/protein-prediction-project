import time

def get_ubiquitination_sites(sequence: str, organism='human', confidence='medium'):
    """
    MOCK FUNCTION for Stage 1.
    This function simulates the output of the UbPred web server for development.
    It returns a fixed list of predicted sites based on the input sequence.

    Args:
        sequence (str): The amino acid sequence of the protein.
        organism (str): The organism (not used in mock).
        confidence (str): The prediction confidence level (not used in mock).

    Returns:
        list: A hardcoded list of tuples for testing Stage 2.
    """
    print("--- RUNNING MOCK STAGE 1 (Server is down for maintenance) ---")
    print("Simulating a 1-second delay for server response...")
    time.sleep(1) # Simulate network delay

    # Find all Lysine (K) positions in the input sequence
    lysine_positions = [i + 1 for i, char in enumerate(sequence) if char == 'K']

    # Return a predefined result ONLY if the sequence has lysines
    if lysine_positions:
        print(f"Found Lysines at positions: {lysine_positions}")
        print("Returning a hardcoded prediction for development.")
        # We will pretend the server predicted the first two lysines are ubiquitinated
        mock_results = []
        if len(lysine_positions) > 0:
            mock_results.append((lysine_positions[0], 'K', 0.91)) # High confidence score
        if len(lysine_positions) > 1:
            mock_results.append((lysine_positions[1], 'K', 0.75)) # Medium confidence score

        return mock_results
    else:
        print("No Lysine (K) residues found in the sequence. Returning empty list.")
        return []

# --- Example Usage ---
if __name__ == "__main__":
    test_sequence_with_lysines = "MIVFWARSVTSLEEAKDPHYPFKPWKVRFSLFEFNYGPYN" \
                                "GREGTRLWRFRWENGEKINTWEGPEGTFGVVFLEENVFNS" \
                                "VVERLEIKKSKGKQNKLDLSNLVIPGVEGIDISETFEVIF" \
                                "TDREYEPVTLTVFQSFKVRWQNLKHMVVFVRIG"

    print("--- Testing with a sequence containing Lysines ---")
    sites = get_ubiquitination_sites(test_sequence_with_lysines)
    if sites:
        print("\nPredicted Ubiquitination Sites (from mock function):")
        for pos, res, scr in sites:
            print(f"  - Position: {pos}, Residue: {res}, Score: {scr}")
    else:
        print("\nNo sites were predicted.")