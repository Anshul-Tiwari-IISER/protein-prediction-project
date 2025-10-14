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

        if not df.empty and 'Sequence' in df.columns:
            sequence = df.iloc[0]['Sequence']
            if pd.notna(sequence):
                return {'accession': accession_id, 'sequence': sequence, 'location': location_category}
        else:
            print(f"Warning: Could not retrieve sequence for {accession_id}. Skipping.")
            return None

    except requests.exceptions.RequestException:
        print(f"Warning: Network error for {accession_id}. Skipping.")
        return None
    return None


if __name__ == "__main__":
    # A much larger, curated list of proteins for a more robust dataset
    protein_locations = {
        'nucleus': ['P04637', 'P08047', 'P10415', 'P15311', 'Q9Y6K1', 'P38398', 'P06400', 'P11387', 'Q04637', 'P27348',
                    'Q13547', 'P16403', 'P20226', 'Q16531', 'O15350'],
        'cytoplasm': ['P60709', 'P62158', 'P07830', 'P41217', 'Q99459', 'P49411', 'P63104', 'P31946', 'Q06830',
                      'P21333', 'P40337', 'P62258', 'Q9UNS2', 'O75367', 'P08238'],
        'mitochondrion': ['P00390', 'P10809', 'P31930', 'Q9Y6L7', 'P05481', 'P00403', 'P00387', 'P30084', 'Q16790',
                          'P22896', 'P36551', 'P08246', 'P13620', 'Q02878', 'P20035'],
        'Golgi': ['Q9Y6N1', 'Q15084', 'Q9Y266', 'P51810', 'O75822', 'Q8N488', 'Q99460', 'Q01850', 'P50406', 'Q9Y5B4',
                  'P53621', 'Q9Y6M9', 'Q15306', 'O00469', 'Q92844'],
        'Endoplasmic': ['Q13085', 'P49756', 'Q14150', 'Q9Y3A4', 'P22681', 'Q96S16', 'Q96Q05', 'Q969X1', 'Q13509',
                        'P11021', 'P04049', 'P51858', 'Q15063', 'P35568', 'P43307']
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
        time.sleep(0.1)

    df = pd.DataFrame(protein_data_list)

    if df.empty:
        print("\nError: Failed to retrieve any protein data.")
    else:
        output_filename = 'location_data.csv'
        df.to_csv(output_filename, index=False)

        print(f"\nSuccessfully created the initial dataset with {len(df)} proteins!")
        print(f"File saved to: {output_filename}")
        print("\nSample of the new data:")
        print(df.head())