import pandas as pd
import requests
from io import StringIO
from tqdm import tqdm
import os
import time


# This function is from your stage2_site_location_predictor.py script
def create_marked_sequence(sequence: str, site_positions: list):
    seq_list = list(sequence)
    for pos in site_positions:
        if 1 <= pos <= len(seq_list) and seq_list[pos - 1] == 'K':
            seq_list[pos - 1] = 'U'
    return "".join(seq_list)


def get_sites_from_uniprot(accession_id: str):
    url = f"https://www.uniprot.org/uniprot/{accession_id}.tsv"
    params = {'fields': 'feature(MODIFIED RESIDUE)', 'query': '*'}
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        tsv_data = StringIO(response.text)
        df = pd.read_csv(tsv_data, sep='\t')

        ubi_sites = []
        if 'Modified residue' in df.columns:
            for item in df['Modified residue'].dropna():
                if 'ubiquitination' in item:
                    position_str = item.split(' ')[1].split(';')[0]
                    if position_str.isdigit():
                        ubi_sites.append(int(position_str))
        return ubi_sites
    except requests.exceptions.RequestException:
        return []


# --- Main Script ---
if __name__ == "__main__":
    input_filename = 'location_data.csv'
    output_filename = 'stage2_localization/marked_location_data.csv'

    # --- Check if we can resume from a previous run ---
    if os.path.exists(output_filename):
        print("Found existing output file. Resuming download.")
        processed_df = pd.read_csv(output_filename)
        # Create a set of accession IDs that are already processed
        processed_accessions = set(processed_df['accession'])
        print(f"Already processed {len(processed_accessions)} proteins.")
    else:
        print("No existing output file found. Starting a new download.")
        processed_df = pd.DataFrame()
        processed_accessions = set()

    original_df = pd.read_csv(input_filename)

    # Filter out the proteins we've already processed
    df_to_process = original_df[~original_df['accession'].isin(processed_accessions)]

    if df_to_process.empty:
        print("All proteins have already been processed. Nothing to do.")
    else:
        print(f"Querying UniProt for {len(df_to_process)} remaining proteins...")

        new_rows = []
        # Use tqdm for a progress bar
        for index, row in tqdm(df_to_process.iterrows(), total=df_to_process.shape[0]):
            accession = row['accession']
            sequence = row['sequence']

            known_sites = get_sites_from_uniprot(accession)
            marked_seq = create_marked_sequence(sequence, known_sites)

            new_row = row.copy()
            new_row['marked_sequence'] = marked_seq
            new_rows.append(new_row)

            # Save progress every 100 proteins
            if (index + 1) % 100 == 0:
                temp_df = pd.DataFrame(new_rows)
                processed_df = pd.concat([processed_df, temp_df], ignore_index=True)
                processed_df.to_csv(output_filename, index=False)
                new_rows = []  # Clear the list after saving

            # Add a small delay to be polite to the server and avoid rate limiting
            time.sleep(0.1)

            # Save any remaining rows
        if new_rows:
            temp_df = pd.DataFrame(new_rows)
            processed_df = pd.concat([processed_df, temp_df], ignore_index=True)
            processed_df.to_csv(output_filename, index=False)

    print(f"\nProcessing complete!")
    print(f"Full dataset saved to: {output_filename}")

    final_df = pd.read_csv(output_filename)
    print("\nSample of the final dataset:")
    print(final_df[['accession', 'sequence', 'marked_sequence', 'location']].head())