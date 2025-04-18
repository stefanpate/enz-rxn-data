import hydra
from omegaconf import DictConfig
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from itertools import product, permutations
from collections import Counter
from typing import Iterable
from rdkit import Chem
from enz_rxn_data.mapping import does_break_cc
import logging
from ergochemics.mapping import (
    operator_map_reaction,
    _m_standardize_reaction,
    rc_to_str,
    rc_to_nest
)

def clean_up_mcsa(mech_atoms: Iterable[Iterable[Iterable[int]]], rxn: str) -> tuple[list[list[int]], str]:
    '''
    De-atom map, remove ions, translate mech atoms from am nums to atom idxs
    '''
    to_rm = {
        'H',
        'Zn',
        'Mg',
        'Fe',
        'Cu',
    }

    lhs = rxn.split(">>")[0].split('.')
    rhs = rxn.split(">>")[1].split('.')
    lhs = [Chem.MolFromSmiles(x) for x in lhs]
    rhs = [Chem.MolFromSmiles(x) for x in rhs]
    lhs_mech_atoms = mech_atoms[0]
    rct_idxs = []
    de_am_smiles = []
    for mol, ams in zip(lhs, lhs_mech_atoms):
        
        # Exclude ions from lhs and rct_idxs
        if mol.GetNumAtoms() == 1 and mol.GetAtomWithIdx(0).GetSymbol() in to_rm:
            continue

        tmp = []
        for atom in mol.GetAtoms():
            if atom.GetAtomMapNum() in ams:
                tmp.append(atom.GetIdx())

            atom.SetAtomMapNum(0)

        rct_idxs.append(tmp)
        de_am_smiles.append(Chem.MolToSmiles(mol))

    # Exclude ions from rhs
    rhs = [
        Chem.MolToSmiles(x) for x in rhs
        if x.GetNumAtoms() > 1 or x.GetAtomWithIdx(0).GetSymbol() not in to_rm
    ]

    rxn = ".".join(de_am_smiles) + ">>" + ".".join(rhs)

    return rct_idxs, rxn

def match_rcts_post_mapping(pre: str, post: str) -> tuple[int]:
    pre_lhs = pre.split(">>")[0].split('.')
    post_lhs = post.split(">>")[0].split('.')
    if len(pre_lhs) != len(post_lhs):
        raise ValueError("Pre and post reaction have different number of reactants")
    
    if len(pre_lhs) == 1:
        return [0]

    pre_rct_idxs = [i for i in range(len(pre_lhs))]
    for perm_idx in permutations(pre_rct_idxs):
        perm = [pre_lhs[i] for i in perm_idx]
        
        if all([pre == post for pre, post in zip(perm, post_lhs)]):
            return perm_idx
        
    raise ValueError("No matching reactants found")

log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="conf", config_name="map_mech_labeled_reactions")
def main(cfg: DictConfig):
    full = pd.read_parquet(Path(cfg.full_rxns)) # TODO: document that full rxns have to be mapped before this file is run
    rule_cts = Counter(full["rule"])

    # Load reactions
    mech_rxns = pd.read_csv(filepath_or_buffer=Path(cfg.mech_rxns), sep=",")
    mech_rxns["mech_atoms"] = mech_rxns["mech_atoms"].apply(lambda x: rc_to_nest(x))

    # Remove atom map numbers & translate mech_atoms from am nums to atom idxs
    mech_atom_idxs = []
    de_am_rxns = []
    for i, row in mech_rxns.iterrows():
        mech_idxs, rxn = clean_up_mcsa(row["mech_atoms"], row["smarts"])
        mech_atom_idxs.append(mech_idxs)
        de_am_rxns.append(rxn)

    mech_rxns["mech_atoms"] = mech_atom_idxs
    mech_rxns["smarts"] = de_am_rxns
    mech_rxns["smarts"] = mech_rxns["smarts"].apply(lambda x: _m_standardize_reaction(x)) # Standardize pre operator mapping to allow matching by smiles after
    
    # Load rules
    rules = pd.read_csv(Path(cfg.filepaths.rules) / "min_rules.csv", sep=",", index_col=0)
    chunksize = len(rules)

    rxn_rule_cart_prod = product(mech_rxns.index, rules.index)
    tasks = [(rxn_id, mech_rxns.loc[rxn_id, "smarts"], rule_id, rules.loc[rule_id, "smarts"]) for rxn_id, rule_id in rxn_rule_cart_prod]
    rxn_ids, rxns, rule_ids, rules = zip(*tasks)
    
    # Map
    with ProcessPoolExecutor() as executor:
        results = list(
            tqdm(
                executor.map(operator_map_reaction, rxns, rules, chunksize=chunksize),
                total=len(tasks)
            )
        )
    
    # Compile results
    columns = ["rxn_id", "smarts", "am_smarts", "rule", "reaction_center", "rule_id"] 
    data = []
    for rxn_id, _, rule, res, rule_id in zip(rxn_ids, rxns, rules, results, rule_ids):
        if res.did_map:
            data.append([rxn_id, res.aligned_smarts, res.atom_mapped_smarts, rule, res.reaction_center, rule_id])

    df = pd.DataFrame(data, columns=columns)

    # Resolved multiple mappings
    selected = []
    for name, group in df.groupby("rxn_id"):
        if len(group) == 1:
            selected.append(group.iloc[0])
        else:
            cc_breaks = group["am_smarts"].apply(does_break_cc)

            if cc_breaks.all() or not cc_breaks.any():
                selected.append(group.loc[group["rule"].map(rule_cts).idxmax()])
            else:
                not_cc = group.loc[~cc_breaks]
                selected.append(not_cc.loc[not_cc["rule"].map(rule_cts).idxmax()])

    selected = pd.DataFrame(selected, columns=df.columns)

    # Compile with other mech rxn cols
    bad_sidxs = []
    bad_ids = []
    for sidx, row in selected.iterrows():
        rxn_id = row['rxn_id']
        post = row["smarts"]
        pre = mech_rxns.loc[rxn_id, "smarts"]
        try:
            perm_idx = match_rcts_post_mapping(pre, post)
            aligned = [mech_rxns.loc[rxn_id]["mech_atoms"][i] for i in perm_idx]
            mech_rxns.at[rxn_id, 'mech_atoms'] = aligned
        except ValueError as e:
            log.info(f"Error matching reactants for {rxn_id}: {e} with smarts {pre} and {post}")
            bad_sidxs.append(sidx)
            bad_ids.append(rxn_id)

    mech_rxns.drop(labels=bad_ids, inplace=True)
    selected.drop(labels=bad_sidxs, inplace=True)
    selected.set_index("rxn_id", inplace=True)
    
    compiled = pd.concat(
        [
            mech_rxns.loc[:, [col for col in mech_rxns.columns if col != "smarts"]],
            selected.loc[:, [col for col in selected.columns if col != 'rxn_id']]
        ],
        axis=1,
        join="inner"
    )

    compiled.reset_index(drop=True, inplace=True)
    compiled = compiled[
        [
            "entry_id", "mechanism_id", "smarts", "am_smarts", "rule", "reaction_center",
            "mech_atoms", "enzyme_name", "uniprot_id", "ec", "rule_id"
        ]
    ]

    compiled["mech_atoms"] = compiled["mech_atoms"].apply(lambda x: rc_to_str([x, [[]]]))
    compiled["reaction_center"] = compiled["reaction_center"].apply(lambda x: rc_to_str(x))

    compiled.to_parquet("mapped_mech_labeled_reactions.parquet") # Save

if __name__ == "__main__":
    main()