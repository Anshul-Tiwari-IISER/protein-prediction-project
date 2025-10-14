import pandas as pd
import requests
from io import StringIO
from tqdm import tqdm
import time


def get_protein_data_from_uniprot(accession_id, location_category):
    """
    Fetches protein sequence from UniProt for a given accession ID.
    """
    url = f"https://rest.uniprot.org/uniprotkb/{accession_id}.tsv"
    params = {'fields': 'accession,sequence'}
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()

        s = StringIO(response.text)
        df = pd.read_csv(s, sep='\t')

        # --- ROBUSTNESS FIX ---
        # Check if the 'Sequence' column exists before trying to access it.
        if not df.empty and 'Sequence' in df.columns:
            sequence = df.iloc[0]['Sequence']
            if pd.notna(sequence):  # Check that the sequence is not empty
                return {'accession': accession_id, 'sequence': sequence, 'location': location_category}
        else:
            # If the column is missing or the dataframe is empty, skip this protein.
            print(f"Warning: Could not retrieve sequence for {accession_id}. Skipping.")
            return None

    except requests.exceptions.RequestException:
        print(f"Warning: Network error for {accession_id}. Skipping.")
        return None
    return None


if __name__ == "__main__":
    # A larger, more reliable list of protein IDs with known locations
    protein_locations = {
        'nucleus': ['P04637', 'P08047', 'P10415', 'P15311', 'Q9Y6K1', 'P38398', 'P06400', 'P11387', 'Q04637', 'P27348'],
        'cytoplasm': ['P60709', 'P62158', 'P07830', 'P41217', 'Q99459', 'P49411', 'P63104', 'P31946', 'Q06830',
                      'P21333'],
        'mitochondrion': ['P00390', 'P10809', 'P31930', 'Q9Y6L7', 'P05481', 'P00403', 'P00387', 'P30084', 'Q16790',
                          'P22896'],
        'Golgi': ['Q9Y6N1', 'Q15084', 'Q9Y266', 'P51810', 'O75822', 'Q8N488', 'Q99460', 'Q01850', 'P50406', 'Q9Y5B4'],
        'Endoplasmic': ['Q13085', 'P49756', 'Q14150', 'Q9Y3A4', 'P22681', 'Q96S16', 'Q96Q05', 'Q969X1', 'Q13509',
                        'P11021']
    }

    all_ids_with_loc = []
    for loc, ids in protein_locations.items():
        for pid in ids:
            all_ids_with_loc.append((pid, loc))

    protein_data_list = []
    print(f"Fetching data for {len(all_ids_with_loc)} proteins from UniProt...")

    for pid, loc in tqdm(all_ids_with_loc):
        data = get_protein_data_from_uniprot(pid, loc)
        if data:
            protein_data_list.append(data)
        time.sleep(0.1)  # Be polite to the server

    df = pd.DataFrame(protein_data_list)

    if df.empty:
        print("\nError: The script ran, but failed to retrieve any protein data.")
    else:
        output_filename = 'location_data.csv'
        df.to_csv(output_filename, index=False)

        print(f"\nSuccessfully created the dataset!")
        print(f"File saved to: {output_filename}")
        print("\nSample of the new data:")
        print(df.head())