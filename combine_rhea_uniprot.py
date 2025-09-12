import pandas as pd
import polars as pl
from pathlib import Path
import re
import hydra
from omegaconf import DictConfig
from tqdm import tqdm
from ergochemics.standardize import standardize_smiles, hash_compound, hash_reaction
from functools import lru_cache
from itertools import chain
import logging
from enz_rxn_data.schemas import known_compounds_schema, enzymes_schema, known_reactions_schema
from rdkit import Chem
from collections import defaultdict

log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="./conf", config_name="combine_rhea_uniprot")
def main(cfg: DictConfig):
    '''
    This function reads a tsv file containing reviewed protein entries from UniProt, and tsv files containing reaction
    SMILES and reaction directions from Rhea, and returns a pandas dataframe containing pertinent enzyme and reaction
    information, respectively.
    '''
    @lru_cache(maxsize=None)
    def std_smi(smiles: str) -> str:
        """
        Standardizes a SMILES string using the standardize_smiles function.
        """
        try:
            return standardize_smiles(smiles, neutralization_method="simple", do_remove_stereo=cfg.rm_stereo, quiet=True)
        except Exception as e:
            log.info(f"Error standardizing SMILES '{smiles}': {e}")
            return None
        
    def std_rxn_side(side: list[str]) -> list[str] | None:
        '''
        Standardize side of reactions, making use of cached standardize_smiles function.
        '''
        std_side = []
        for smi in side:
            if smi == '[H+]':
                continue  # Skip protons
            
            _std_smi = std_smi(smi)
            
            if _std_smi is None:
                log.info(f"Unable to standardize SMILES: {smi}")
                return None
        
            std_side.append(_std_smi)
        
        return std_side
    
    # Get compound smiles to names
    log.info("Collecting compounds...")
    chebi2name = pd.read_csv(Path(cfg.chebi2name), sep='\t', header=None)
    chebi2name.columns = ['CHEBI_ID', 'NAME']
    chebi2smiles = pd.read_csv(Path(cfg.chebi2smiles), sep='\t', header=None)
    chebi2smiles.columns = ['CHEBI_ID', 'SMILES']
    smiles2names = pd.merge(chebi2smiles, chebi2name, on='CHEBI_ID', how='inner')
    smiles2names["SMILES"] = smiles2names["SMILES"].apply(std_smi)
    smiles2names.drop_duplicates(subset='SMILES', inplace=True)
    
    # Read in UniProt protein entries
    enz_df = pd.read_csv(Path(cfg.uniprot), sep='\t')
    enz_df = enz_df.rename(columns = {'Entry': 'UniProt_Entry'})

    # Filter out subunits
    prev_len = len(enz_df)
    enz_df['Protein names'] = enz_df['Protein names'].fillna('unknown')
    enz_df['subunit'] = enz_df['Protein names'].str.contains('subunit')
    log.info(f"Identified {enz_df['subunit'].sum()} subunits in UniProt entries.")

    # Read in Rhea SMILES and assign appropriate column names
    rxn_smiles_df = pd.read_csv(Path(cfg.rhea_smiles), sep='\t', header=None)
    rxn_smiles_df.columns = ['RHEA_ID', 'RXN_SMILES']

    log.info("Standardizing reaction SMILES...")
    tmp = []
    for i in tqdm(range(len(rxn_smiles_df))):
        rxn = rxn_smiles_df.loc[i, 'RXN_SMILES']
        lhs, rhs = [side.split('.') for side in rxn.split('>>')]
        std_lhs = std_rxn_side(lhs)
        
        if std_lhs is None:
            continue  # Skip reactions that failed to standardize on the left-hand side
        
        std_rhs = std_rxn_side(rhs)

        if std_rhs is None:
            continue # Skip reactions that failed to standardize on the right-hand side

        tmp.append('.'.join(sorted(std_lhs)) + '>>' + '.'.join(sorted(std_rhs)))
        
    rxn_smiles_df['RXN_SMILES'] = tmp

    # Read in rhea directions
    rhea_directions = pd.read_csv(Path(cfg.rhea_directions), sep='\t')

    # Funnel all rhea ids to single working index
    any_rhea_to_working_idx = {}
    for i, row in rhea_directions.iterrows():
        any_rhea_to_working_idx[row['RHEA_ID_MASTER']] = i
        any_rhea_to_working_idx[row['RHEA_ID_LR']] = i
        any_rhea_to_working_idx[row['RHEA_ID_RL']] = i
        any_rhea_to_working_idx[row['RHEA_ID_BI']] = i

    rxn_smiles_df["WORKING_IDX"] = rxn_smiles_df['RHEA_ID'].apply(lambda x: any_rhea_to_working_idx.get(x))

    # Collect RHEA IDs iterating over Uniprot entries
    log.info("Merging rhea and uniprot entries...")
    matching_enz_ids = defaultdict(set)
    for _, row in tqdm(enz_df.iterrows(), total=len(enz_df)):
        catalytic_activity = row['Catalytic activity']
        uniprot_entry = row['UniProt_Entry']
        if isinstance(catalytic_activity, str):
            match = re.findall(r'RHEA:(\d{1,6})', catalytic_activity)
            for rhea_id in match:
                matching_enz_ids[any_rhea_to_working_idx[int(rhea_id)]].add(uniprot_entry)
    
    log.info(f"Found {sum(len(v) for v in matching_enz_ids.values())} enzyme-reaction pairs in Uniprot entries.")
    rxn_smiles_df['ENZYME_ID'] = rxn_smiles_df['WORKING_IDX'].apply(lambda x: list(matching_enz_ids[x]) if x in matching_enz_ids else [])
    all_matched_enz_ids = set(chain(*[v for v in matching_enz_ids.values()]))
    enz_df = enz_df[enz_df["UniProt_Entry"].isin(all_matched_enz_ids)] # Just keep enzymes that have reactions

    # Fold up redundant smiles, e.g., had differed by stereochemistry
    rxn_smiles_df = rxn_smiles_df.groupby("RXN_SMILES").agg(
        {
            "ENZYME_ID": lambda x: sum([lst for lst in x if isinstance(lst, list)], []), 
            "RHEA_ID": list
        }
    ).reset_index()
    rxn_smiles_df["REV_RXN_SMILES"] = rxn_smiles_df["RXN_SMILES"].apply(lambda x: ">>".join(x.split(">>")[::-1]))
    rxn_smiles_df = rxn_smiles_df[rxn_smiles_df["RXN_SMILES"] != rxn_smiles_df["REV_RXN_SMILES"]] # Drop non-reactions, e.g., epimerization, transport
    rxn_smiles_df["UNIQUE_ID"] = rxn_smiles_df["RXN_SMILES"].apply(hash_reaction)
    rxn_smiles_df["REVERSE_ID"] = rxn_smiles_df["REV_RXN_SMILES"].apply(lambda x: rxn_smiles_df.loc[rxn_smiles_df["RXN_SMILES"] == x, "UNIQUE_ID"].values[0])
    
    # Leave room to grow into other databases beyond rhea
    rxn_smiles_df['DB_IDs'] = rxn_smiles_df['RHEA_ID'].apply(lambda x: [f"RHEA:{elt}" for elt in x])

    known_reactions = pl.DataFrame(
        {
            "id": rxn_smiles_df['UNIQUE_ID'],
            "smarts": rxn_smiles_df['RXN_SMILES'],
            "enzymes": rxn_smiles_df['ENZYME_ID'].apply(lambda x: list(set(x)) if isinstance(x, list) else []),
            "reverse": rxn_smiles_df['REVERSE_ID'],
            "db_ids": rxn_smiles_df['DB_IDs'].apply(lambda x: list(set(x)) if isinstance(x, list) else []),
        },
        schema=known_reactions_schema,
    )

    known_enzymes = pl.DataFrame(
        {
            "id": enz_df['UniProt_Entry'],
            "sequence": enz_df['Sequence'],
            "existence": enz_df['Protein existence'].apply(lambda x: x if isinstance(x, str) else 'Uncertain'),
            "reviewed": enz_df['Reviewed'].apply(lambda x: 'reviewed' if x else 'unreviewed'),
            "ec": enz_df['EC number'],
            "organism": enz_df['Organism'],
            "name": enz_df['Protein names'],
            "subunit": enz_df['subunit'],
        },
        schema=enzymes_schema,
    )

    # Collect only compounds appearing in reactions
    cpds_in_rxns = set()
    for row in known_reactions.iter_rows(named=True):
        rxn_smarts = row['smarts']
        for cpd in chain(*[side.split('.') for side in rxn_smarts.split('>>')]):
            cpds_in_rxns.add(cpd)

    smiles2names = smiles2names[smiles2names['SMILES'].isin(cpds_in_rxns)]
    smiles2names.sort_values(by='SMILES', inplace=True)

    known_compounds = pl.DataFrame(
        {
            "id": smiles2names["SMILES"].apply(hash_compound),
            "smiles": smiles2names["SMILES"],
            "name": smiles2names["NAME"].apply(lambda x: x.strip()),
            "chebi_id": smiles2names["CHEBI_ID"],
            "n_atoms": smiles2names["SMILES"].apply(lambda x: Chem.MolFromSmiles(x).GetNumAtoms()),
        },
        schema=known_compounds_schema,
    )

    log.info(f"Saving {len(known_compounds)} compounds, {len(known_reactions)} reactions, & {len(known_enzymes)} enzymes...")
    known_compounds.write_parquet('known_compounds.parquet')
    known_reactions.write_parquet('known_reactions.parquet')
    known_enzymes.write_parquet('known_enzymes.parquet')

if __name__ == '__main__':
    main()