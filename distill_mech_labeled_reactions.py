import hydra
from omegaconf import DictConfig
from pathlib import Path
import json
import pandas as pd
import numpy as np
from itertools import chain
from functools import reduce
from rdkit import Chem
from ast import literal_eval
import logging
from collections import defaultdict
from enz_rxn_data.mechanism import get_overall_reaction
from ergochemics.standardize import standardize_mol
from ergochemics.mapping import rc_to_str

def is_balanced(lhs: list[Chem.Mol], rhs: list[Chem.Mol]) -> bool:
    if len(lhs) == 0 or len(rhs) == 0:
        return False
    
    lhs_atoms = defaultdict(int)
    rhs_atoms = defaultdict(int)

    for mol in lhs:
        for atom in mol.GetAtoms():
            lhs_atoms[atom.GetSymbol()] += 1

    for mol in rhs:
        for atom in mol.GetAtoms():
            rhs_atoms[atom.GetSymbol()] += 1

    return lhs_atoms == rhs_atoms

def match_overall_candidates(lhs: list[Chem.Mol], rhs: list[Chem.Mol], lhs_candidate: list[Chem.Mol], rhs_candidate: list[Chem.Mol]) -> tuple[list[Chem.Mol], list[Chem.Mol]]:
    '''
    Matches the overall reaction candidates to the lhs and rhs of the mechanistic reaction.
    
    Args
    ----
    lhs: list[Chem.Mol]
        Full lhs of reaction gotten from elementary steps
    rhs: list[Chem.Mol]
        Full rhs of reaction gotten from elementary steps
    lhs_candidate: list[Chem.Mol]
        Putative overall reaction lhs
    rhs_candidate: list[Chem.Mol]
        Puative overall reaction rhs
    
    Returns
    -------
    tuple[list[Chem.Mol], list[Chem.Mol]]
        Tuple containing the matched left-hand side and right-hand side molecules.
        Returns empty lists if a full match cannot be made.
    '''
    tasks = list(zip([lhs_candidate, rhs_candidate], [lhs, rhs]))
    matches = [[], []]
    for k, (candidate_side, side) in enumerate(tasks):
        remaining_cidxs = set([i for i in range(len(candidate_side))])
        remaining_idxs = set([i for i in range(len(side))])
        for cidx, cmol in enumerate(candidate_side):
            for idx in remaining_idxs:
                mol = side[idx]
                
                if cmol.GetNumAtoms() != mol.GetNumAtoms(): # Cardinality different
                    continue

                ss_match = cmol.GetSubstructMatch(mol)
                if len(ss_match) == cmol.GetNumAtoms(): # Mols match
                    matches[k].append(mol)
                    remaining_cidxs.remove(cidx)
                    remaining_idxs.remove(idx)
                    break
                
                cmol_std = standardize_mol(cmol, quiet=True)
                mol_std = standardize_mol(mol, quiet=True)
                ss_match = cmol_std.GetSubstructMatch(mol_std)
                if len(ss_match) == cmol_std.GetNumAtoms(): # Standardized mols match
                    matches[k].append(mol)
                    remaining_cidxs.remove(cidx)
                    remaining_idxs.remove(idx)
                    break
        
        if len(remaining_cidxs) > 0:
            return [], []
        
    return matches
    
def remove_unchanged_molecules(lhs: list[str], rhs: list[str]) -> tuple[list[Chem.Mol], list[Chem.Mol]]:
    '''
    Tries to distill down to overall reaction by removing unchanged molecules, i.e., residues and 
    cofactors.

    Args
    ----
    lhs: list[str]
        List of SMILES strings for the left-hand side of the reaction.
    rhs: list[str]
        List of SMILES strings for the right-hand side of the reaction.
    Returns
    -------
    tuple[list[Chem.Mol], list[Chem.Mol]]
        Tuple of lists containing the RDKit molecules for the left-hand side and right-hand side of the reaction.
    '''
    lhs_smi_pos = defaultdict(list)
    rhs_smi_pos = defaultdict(list)

    for i, smi in enumerate(lhs):
        lhs_smi_pos[
            rm_amns(Chem.MolFromSmiles(smi))
        ].append(i)
        
    for i, smi in enumerate(rhs):
        rhs_smi_pos[
            rm_amns(Chem.MolFromSmiles(smi))
        ].append(i)

    # Filter out smarts that are in both sides with equal cardinalities
    unequal_cards = set()
    lhs_keep = set()
    for k, v in lhs_smi_pos.items():
        if k not in rhs_smi_pos:
            for i in v:
                lhs_keep.add(i)
        elif len(v) != len(rhs_smi_pos[k]):
            unequal_cards.add(k)
    
    rhs_keep = set()
    for k, v in rhs_smi_pos.items():
        if k not in lhs_smi_pos:
            for i in v:
                rhs_keep.add(i)
        elif len(v) != len(lhs_smi_pos[k]):
            unequal_cards.add(k)

    if len(lhs_keep) == 0 or len(rhs_keep) == 0:
        return [], []

    lhs_mols = [Chem.MolFromSmiles(lhs[i]) for i in lhs_keep]
    rhs_mols = [Chem.MolFromSmiles(rhs[i]) for i in rhs_keep]

    lhs_amns = list(
        chain(
            *[[atom.GetAtomMapNum() for atom in mol.GetAtoms()] for mol in lhs_mols]
        )
    )
    rhs_amns = list(
        chain(
            *[[atom.GetAtomMapNum() for atom in mol.GetAtoms()] for mol in rhs_mols]
        )
    )

    # Keep only those from unequal_cards that have have amn on other side of distinct mols, i.e., those in lhs/rhs mols
    for i in unequal_cards:

        for j in lhs_smi_pos[i]:
            mol = Chem.MolFromSmiles(lhs[j])
            for atom in mol.GetAtoms():
                if atom.GetAtomMapNum() in rhs_amns:
                    lhs_mols.append(mol)
                    break

        for j in rhs_smi_pos[i]:
            mol = Chem.MolFromSmiles(rhs[j])
            for atom in mol.GetAtoms():
                if atom.GetAtomMapNum() in lhs_amns:
                    rhs_mols.append(mol)
                    break

    return lhs_mols, rhs_mols 

def rm_amns(mol: Chem.Mol) -> str:
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(mol)

def is_strictly_balanced(lhs: list[Chem.Mol], rhs: list[Chem.Mol]) -> bool:
    '''
    Check if the reaction is strictly balanced, i.e., all atom map numbers are conserved.
    '''
    lhs_elements = {f"{atom.GetSymbol()}_{atom.GetAtomMapNum()}" for mol in lhs for atom in mol.GetAtoms()}
    rhs_elements = {f"{atom.GetSymbol()}_{atom.GetAtomMapNum()}" for mol in rhs for atom in mol.GetAtoms()}
    
    return len(lhs_elements ^ rhs_elements) == 0

def collect_rhea_rxns(smiles_fp: Path, directions_fp: Path, ec_fp: Path) -> dict[int, set[str]]:
    rhea_rxn_smarts = pd.read_csv(smiles_fp, sep="\t", header=None)
    rhea_ec = pd.read_csv(ec_fp, sep="\t")
    rhea_directions = pd.read_csv(directions_fp, sep="\t")
    rhea_rxn_smarts.columns = ['ID', 'SMARTS']
    rhea_ec.columns = ["RHEA_ID", "DIRECTION", "MASTER_ID", "EC"]
    rhea_directions.columns = ["RHEA_ID", "RHEA_ID_LR", "RHEA_ID_RL", "RHEA_ID_BI"]
    rhea = rhea_ec.join(other=rhea_directions.set_index("RHEA_ID"), on="RHEA_ID", how="inner")
    rhea = rhea.join(other=rhea_rxn_smarts.set_index("ID"), on="RHEA_ID_LR", how="left")
    rhea = rhea.join(other=rhea_rxn_smarts.set_index("ID"), on="RHEA_ID_RL", how="left", rsuffix="_RL")
    
    alt_rxns = defaultdict(set)
    for _, row in rhea.iterrows():
        for smarts in [row["SMARTS"], row["SMARTS_RL"]]:
            smarts = row["SMARTS"]
            
            if not pd.isna(smarts):
                alt_rxns[row["EC"]].add(smarts)

    return alt_rxns

log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="conf", config_name="distill_mech_labeled_reactions")
def main(cfg: DictConfig):
    min_amn = lambda mol: min(atom.GetAtomMapNum() for atom in mol.GetAtoms() if atom.GetAtomMapNum() > 0)
    sms_std = lambda smi: Chem.MolToSmiles(standardize_mol(Chem.MolFromSmiles(smi), quiet=True), ignoreAtomMapNumbers=True)
    is_H_ion = lambda mol : all([atom.GetAtomicNum() == 1 for atom in mol.GetAtoms()]) and len(mol.GetAtoms()) == 1

    rhea_smiles_fp = Path(cfg.filepaths.raw_data) / "pathway" / "rhea-reaction-smiles.tsv"
    rhea_directions_fp = Path(cfg.filepaths.raw_data) / "pathway" / "rhea-directions.tsv"
    rhea2ec_fp = Path(cfg.filepaths.raw_data) / "pathway" / "rhea2ec.tsv"

    alt_rxns = collect_rhea_rxns(rhea_smiles_fp, rhea_directions_fp, rhea2ec_fp)

    # Load mcsa entries
    entries = {}
    for i in cfg.entry_batches:
        with open(Path(cfg.filepaths.raw_mcsa) / f"entries_{i}.json", "r") as f:
            entries = {**entries, **json.load(f)}

    # Load mech labeled reactions
    mech_rxns = pd.read_csv(filepath_or_buffer=Path(cfg.mech_rxns), sep=",")
    mech_rxns["mech_atoms"] = mech_rxns["mech_atoms"].apply(literal_eval)

    # Main loop
    data = []
    for _, row in mech_rxns.iterrows():
        # Full mech labeled reaction
        lhs, rhs = [
            [sms_std(smi) for smi in side.split('.')]
            for side in row["smarts"].split('>>')
        ]

        mcsa_entry = entries[str(row["entry_id"])]

        # Candidate overall reactions (distilled) from M-CSA / Rhea
        lhs_candidate, rhs_candidate = get_overall_reaction(mcsa_entry['reaction']['compounds'], Path(cfg.filepaths.raw_mcsa) / "mols")
        lhs_candidate = [elt for elt in lhs_candidate if not is_H_ion(elt)] # Forgive H ions
        rhs_candidate = [elt for elt in rhs_candidate if not is_H_ion(elt)]

        overall_rxn_candidates = []
        if is_balanced(lhs_candidate, rhs_candidate): # Check if overall candidates balanced
            overall_rxn_candidates = [(lhs_candidate, rhs_candidate)]
        else: # Check rhea for overall rxn candidates
            for sma in alt_rxns[mcsa_entry['reaction']['ec']]:
                lhs_candidate, rhs_candidate = [[Chem.MolFromSmiles(smi) for smi in side.split(".")] for side in sma.split(">>")]
                lhs_candidate = [elt for elt in lhs_candidate if not is_H_ion(elt)]
                rhs_candidate = [elt for elt in rhs_candidate if not is_H_ion(elt)]
                if is_balanced(lhs_candidate, rhs_candidate):
                    overall_rxn_candidates.append((lhs_candidate, rhs_candidate))
        
        # Add the overall candidate reverses
        rev = []
        for lhs_candidate, rhs_candidate in overall_rxn_candidates:
            rev.append((rhs_candidate, lhs_candidate))
        overall_rxn_candidates.extend(rev)
        
        # First try to match overall reaction candidates to mech labeled reaction
        matched = False
        for lhs_candidate, rhs_candidate in overall_rxn_candidates:
            lhs_mols, rhs_mols = match_overall_candidates(
                [Chem.MolFromSmiles(elt) for elt in lhs],
                [Chem.MolFromSmiles(elt) for elt in rhs],
                lhs_candidate,
                rhs_candidate
            )

            if len(lhs_mols) != 0 and len(rhs_mols) != 0:
                matched = True
                break

        if not matched: # Matching failed
            log.info(f"Could not match overall reaction candidates to mech labeled reaction for entry {(row['entry_id'], row['mechanism_id'])}")
            lhs_mols, rhs_mols = remove_unchanged_molecules(lhs, rhs) # Try to distill down

            if len(lhs_mols) == 0 or len(rhs_mols) == 0:
                log.info(f"Skipping entry {(row['entry_id'], row['mechanism_id'])} because no molecules left after removing unchanged molecules")
                continue

            if not is_strictly_balanced(lhs_mols, rhs_mols):
                log.info(f"Skipping entry {(row['entry_id'], row['mechanism_id'])} because not strictly balanced after distillation")
                continue
        elif not is_strictly_balanced(lhs_mols, rhs_mols): # Matching did not yield strictly balanced reaction
            lhs_mols, rhs_mols = remove_unchanged_molecules(lhs, rhs) # Try to distill down

            if not is_strictly_balanced(lhs_mols, rhs_mols): # Still not strictly balanced
                log.info(f"Skipping entry {(row['entry_id'], row['mechanism_id'])} because not strictly balanced after distillation")
                continue

        # Create combined molecules
        lhs = reduce(Chem.CombineMols, lhs_mols)
        rhs = reduce(Chem.CombineMols, rhs_mols)

        # Get reaction center amns
        lhs_amns = [atom.GetAtomMapNum() for atom in lhs.GetAtoms()]
        rhs_amns = [atom.GetAtomMapNum() for atom in rhs.GetAtoms()]
        lhs_perm = sorted([i for i in range(len(lhs_amns))], key=lambda x: lhs_amns[x])
        rhs_perm = sorted([i for i in range(len(rhs_amns))], key=lambda x: rhs_amns[x])
        lhs_A = Chem.GetAdjacencyMatrix(lhs, useBO=True)[lhs_perm][:, lhs_perm]
        rhs_A = Chem.GetAdjacencyMatrix(rhs, useBO=True)[rhs_perm][:, rhs_perm]
        rc = np.abs(lhs_A - rhs_A).sum(axis=1).nonzero()[0].tolist()
        rc_amns = [lhs_amns[lhs_perm[i]] for i in rc] # Translate back to amns; either lhs or rhs will do

        # Convert from amns in aidxs
        tmp_rc = [[], []]
        tmp_mech = [[], []]
        mech_amns = row['mech_atoms']
        for i, side in enumerate([lhs_mols, rhs_mols]):
            for mol in side:
                tmp_rc[i].append([])
                tmp_mech[i].append([])
                for atom in mol.GetAtoms():
                    amn = atom.GetAtomMapNum()
                    if amn in rc_amns:
                        tmp_rc[i][-1].append(atom.GetIdx())
                    elif amn in mech_amns:
                        tmp_mech[i][-1].append(atom.GetIdx())

        std_rxn = [[], []]
        std_am_rxn = [[], []]
        rc_aidxs = [[], []]
        mech_aidxs = [[], []]
        for i, side in enumerate([lhs_mols, rhs_mols]):
            amn_order = sorted([j for j in range(len(side))], key=lambda x: min_amn(side[x]))
            for j in amn_order:

                if len(tmp_rc[i][j]) == 0: # No reaction center atoms
                    continue

                std_am_rxn[i].append(Chem.MolToSmiles(side[j], ignoreAtomMapNumbers=True))
                std_rxn[i].append(rm_amns(side[j]))
                rc_aidxs[i].append(tmp_rc[i][j])
                mech_aidxs[i].append(tmp_mech[i][j])

        std_rxn = ".".join(std_rxn[0]) + ">>" + ".".join(std_rxn[1])
        std_am_rxn = ".".join(std_am_rxn[0]) + ">>" + ".".join(std_am_rxn[1])

        if std_rxn == '>>':
            log.info(f"No molecules remain after removing non-rc-bearing molecules {(row['entry_id'], row['mechanism_id'])}")
            continue

        # Reported direction
        data.append(
            [
                row["entry_id"], row["mechanism_id"], std_rxn, std_am_rxn,
                rc_aidxs, mech_aidxs, row["enzyme_name"], row["uniprot_id"], row["ec"], True
            ]
        )

        # Reverses
        data.append(
            [
                row["entry_id"], row["mechanism_id"], ">>".join(std_rxn.split('>>')[::-1]),
                ">>".join(std_am_rxn.split('>>')[::-1]), rc_aidxs[::-1], mech_aidxs[::-1],
                row["enzyme_name"], row["uniprot_id"], row["ec"], False
            ]
        )

    columns = ["entry_id", "mechanism_id", "smarts", "am_smarts", "reaction_center", "mech_atoms", "enzyme_name", "uniprot_id", "ec", "reported_direction"]
    distilled = pd.DataFrame(data, columns=columns)
    distilled["mech_atoms"] = distilled["mech_atoms"].apply(rc_to_str)
    distilled["reaction_center"] = distilled["reaction_center"].apply(rc_to_str)
    distilled.to_parquet("distilled_mech_reactions.parquet") # Save

if __name__ == "__main__":
    main()