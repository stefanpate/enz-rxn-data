import pandas as pd
from pathlib import Path
import re

def main(uniprot_protein_entries: Path, rhea_smiles: Path, rhea_directions: Path):
    '''
    This function reads a tsv file containing reviewed protein entries from UniProt, and tsv files containing reaction
    SMILES and reaction directions from Rhea, and returns a pandas dataframe containing pertinent enzyme and reaction
    information, respectively.

    Args
    ----
    uniprot_protein_entries: Path
        Path to the tsv file containing reviewed protein entries from UniProt
    rhea_smiles: Path
        Path to the tsv file containing reaction SMILES from Rhea
    rhea_directions: Path
        Path to the tsv file containing reaction directions from Rhea

    Returns
    -------
    '''
    #Read in UniProt protein entries
    enz_df = pd.read_csv(uniprot_protein_entries, sep='\t')
    enz_df = enz_df.rename(columns = {'Entry': 'UniProt_Entry'})

    #Read in Rhea SMILES and assign appropriate column names
    rxn_smiles_df = pd.read_csv(rhea_smiles, sep='\t', header=None)
    rxn_smiles_df.columns = ['RHEA_ID', 'RXN_SMILES']

    #Read in Rhea directions
    rxn_directions_df = pd.read_csv(rhea_directions, sep='\t')

    #Assign unique IDs to Rhea IDs
    unique_id = {rhea_id: int(idx) for idx, rhea_id in enumerate(rxn_smiles_df['RHEA_ID'].unique(), start=1)}
    
    #Map the unique IDs
    rxn_smiles_df['UNIQUE_ID'] = rxn_smiles_df['RHEA_ID'].map(unique_id)
    rxn_directions_df['UNIQUE_ID_LR'] = rxn_directions_df['RHEA_ID_LR'].map(unique_id).fillna(-1).astype(int)
    rxn_directions_df['UNIQUE_ID_RL'] = rxn_directions_df['RHEA_ID_RL'].map(unique_id).fillna(-1).astype(int)

    #Create a dictionary to link the forward and reverse unique ids
    rxn_directions = rxn_directions_df.set_index('RHEA_ID_MASTER')[['UNIQUE_ID_LR', 'UNIQUE_ID_RL']].to_dict(orient = 'index')

    #List the linked forward and reverse ids for each reaction unique ID
    reverse_id_mapping = {}
    for master_id, ids in rxn_directions.items():
        reverse_id_mapping[ids['UNIQUE_ID_LR']] = ids['UNIQUE_ID_RL']
        reverse_id_mapping[ids['UNIQUE_ID_RL']] = ids['UNIQUE_ID_LR']

    rxn_smiles_df['REVERSE_ID'] = rxn_smiles_df['UNIQUE_ID'].map(reverse_id_mapping)

    #Create a column that includes database ids
    #Append list with other non-Rhea database id entries if needed
    rxn_smiles_df['DB_IDs'] = rxn_smiles_df['RHEA_ID'].apply(lambda x: [f'RHEA:{x}'])

    #Collect RHEA IDs iterating over Uniprot entries
    matching_enz_ids = []
    for _, row in enz_df.iterrows():
        catalytic_activity = row['Catalytic activity']
        uniprot_entry = row['UniProt_Entry']
        if isinstance(catalytic_activity, str):
            match = re.findall(r'RHEA:(\d{5})', catalytic_activity)
            for rhea_id in match:
                matching_enz_ids.append({'RHEA_ID': rhea_id, 'ENZYME_ID': uniprot_entry})
    
    matching_entries_df = pd.DataFrame(matching_enz_ids, columns=['RHEA_ID', 'ENZYME_ID'])
    enz_match_df = matching_entries_df.groupby('ENZYME_ID')['RHEA_ID'].apply(list).reset_index()
    rxn_match_df = matching_entries_df.groupby('RHEA_ID')['ENZYME_ID'].apply(list).reset_index()

    #Add enzyme IDs to the reaction dataframe and add the reaction IDs to the enzyme dataframe
    rxn_smiles_df = pd.concat([rxn_smiles_df, rxn_match_df], axis = 1)
    enz_df = pd.merge(enz_df, enz_match_df, left_on='UniProt_Entry', right_on='ENZYME_ID', how='left')

    #Only keep relevant columns
    rxn_smiles_df.drop(columns = ['RHEA_ID'], inplace = True)
    rxn_smiles_df = rxn_smiles_df[['UNIQUE_ID', 'RXN_SMILES', 'ENZYME_ID', 'REVERSE_ID', 'DB_IDs']]

    enz_df.drop(columns = ['Catalytic activity', 'ENZYME_ID'], inplace = True)

    #Export dataframs as parquet files to interim data folder
    enz_df.to_parquet('/home/kroberts/enz-rxn-data/data/interim/uniprot_rhea/protein_entries.parquet', engine = 'pyarrow', index = False)
    rxn_smiles_df.to_parquet('/home/kroberts/enz-rxn-data/data/interim/uniprot_rhea/reaction_entries.parquet', engine = 'pyarrow', index = False)

    return

if __name__ == '__main__':
    # TODO: Fill in the corresponding paths and hand then to main
    uniprot_protein_entries = Path("/home/kroberts/enz-rxn-data/data/raw/uniprot/uniprotkb_reviewed_true_2025_04_01.tsv")
    rhea_smiles = Path("/home/kroberts/enz-rxn-data/data/raw/rhea/rhea-reaction-smiles.tsv")
    rhea_directions = Path("/home/kroberts/enz-rxn-data/data/raw/rhea/rhea-directions.tsv")

    # Call the main function
    main(uniprot_protein_entries, rhea_smiles, rhea_directions)