import hydra
import logging
from omegaconf import DictConfig
from pathlib import Path
import json
from copy import deepcopy
from itertools import chain
from typing import Iterable
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem import rdChemReactions
import pandas as pd
from ergochemics.standardize import standardize_mol
from ergochemics.mapping import rc_to_str
from enz_rxn_data.mechanism import (
    parse_mrv,
    construct_mols,
    get_overall_reaction,
    step
)

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

def standardize_de_atom_map(mol: Chem.Mol) -> str:
    mol = deepcopy(mol)
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    
    mol = standardize_mol(mol, quiet=True)

    return Chem.MolToSmiles(mol)

def translate(entering_rcts: list[Chem.Mol], prev_products: tuple[Chem.Mol], lhs: Iterable[Chem.Mol]) -> tuple[dict[str, str], tuple[Chem.Mol]]:
    '''
    Finds correspondence between atoms in entering_rcts & prev_products
    with the left-hand side of the elementary step.

    Args:
        entering_rcts: List of reactants entering this elementary step
        prev_products: Products from previous elementary step
        lhs: Left-hand side of the elementary step
    Returns:
        one_step_translation: Maps old atom index to new atom index
        imt_rcts: Intermediate reactants, ordered as they match lhs
    '''
    one_step_translation = {} # Maps old atom index to new atom index
    imt_rcts = [None for _ in range(len(lhs))] # Properly arranged intermediate reactants
    candidates = entering_rcts + prev_products
    remaining_candidates = {i for i in range(len(candidates))}
    for i, query in enumerate(lhs):
         for j, mol in enumerate(candidates):
            
            if j not in remaining_candidates: # Already used
                continue
            
            ss_match = mol.GetSubstructMatch(query)
            if len(ss_match) == mol.GetNumAtoms(): # Exact match
                imt_rcts[i] = query # Smooth over small differences

                for query_idx, mol_idx in enumerate(ss_match):
                    atom  = mol.GetAtomWithIdx(mol_idx)
                    atom_prop_dict = atom.GetPropsAsDict()
                    
                    if 'pre_mech' in atom_prop_dict:
                        old_aidx = atom_prop_dict['pre_mech']
                    elif atom.GetAtomMapNum() != 0:
                        old_aidx = atom.GetAtomMapNum()
                    else:
                        old_aidx = atom_prop_dict['old_mapno']
                    
                    new_aidx = query.GetAtomWithIdx(query_idx).GetAtomMapNum()
                    one_step_translation[old_aidx] = new_aidx

                remaining_candidates.remove(j)
                break

    # Molecules not required this step
    for i in remaining_candidates:
        for atom in candidates[i].GetAtoms():
            atom_prop_dict = atom.GetPropsAsDict()
                    
            if atom.GetAtomMapNum() != 0:
                old_aidx = atom.GetAtomMapNum()
            else:
                old_aidx = atom_prop_dict['old_mapno']
            
            one_step_translation[old_aidx] = None

    # Fill in the rest, e.g., residues not listed in overall reaction
    for i, patt in enumerate(lhs):
        if imt_rcts[i] is None:
            imt_rcts[i] = patt

    return one_step_translation, tuple(imt_rcts)

def transform(lhs: list[Chem.Mol], rhs: list[Chem.Mol], imt_rcts: list[Chem.Mol]) -> list[Chem.Mol]:
    '''
    Creates an rdkit reaction from the elementary step and applies it to the intermediate reactants.

    Args:
        lhs: Left-hand side of the elementary step
        rhs: Right-hand side of the elementary step
        imt_rcts: Intermediate reactants, ordered as they match lhs
    Returns:
        outputs: Products from applying the elementary step to the intermediate reactants
    '''
    rule = ".".join([Chem.MolToSmarts(mol) for mol in lhs]) + ">>" + ".".join([Chem.MolToSmarts(mol) for mol in rhs])
    op = rdChemReactions.ReactionFromSmarts(rule, useSmiles=False)
    outputs = op.RunReactants(imt_rcts)
    
    # there are still cases that hit this like mcsa reports coordinate bonds
    # as single covalent bond, rdkit converts to dative automatically (e.g., entry 102)
    if len(outputs) == 0:
        op = rdChemReactions.ReactionFromSmarts(rule, useSmiles=True)
        outputs = op.RunReactants(imt_rcts)

    smi_outputs = {tuple([Chem.MolToSmiles(mol) for mol in output]) for output in outputs}

    if len(smi_outputs) != 1:
        log.info(f"Multiple unique outputs for rule {rule} and reactants {[Chem.MolToSmiles(mol) for mol in imt_rcts]}")

    return list(outputs[0])

log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="conf", config_name="label_mechanistic_subgraphs")
def main(cfg: DictConfig):
    msm = lambda x : [Chem.MolFromSmiles(Chem.MolToSmiles(mol)) for mol in x] # TODO: Do I need this?
    is_hydrogen = lambda mol : all([atom.GetAtomicNum() == 1 for atom in mol.GetAtoms()]) and len(mol.GetAtoms()) == 1
    
    # Load entries
    entries = {}
    for i in cfg.entry_batches:
        with open(Path(cfg.filepaths.raw_mcsa) / f"entries_{i}.json", "r") as f:
            entries = {**entries, **json.load(f)}

    # Gather alternative overall reactions
    rhea_rxn_smarts = pd.read_csv(Path(cfg.filepaths.raw_data) / "pathway" / "rhea-reaction-smiles.tsv", sep="\t", header=None)
    rhea_ec = pd.read_csv(Path(cfg.filepaths.raw_data) / "pathway" / "rhea2ec.tsv", sep="\t")
    rhea_directions = pd.read_csv(Path(cfg.filepaths.raw_data) / "pathway" / "rhea-directions.tsv", sep="\t")
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

    mech_labeled_reactions = []
    columns = ['entry_id', 'mechanism_id', 'smarts', 'mech_atoms', "enzyme_name", "uniprot_id", "ec"]
    for entry_id, entry in {'722': entries['722']}.items():#entries.items():
        reaction_entry = entry['reaction']
        tmp_overall_lhs, tmp_overall_rhs = get_overall_reaction(reaction_entry['compounds'], Path(cfg.filepaths.raw_mcsa) / "mols")
        tmp_overall_lhs = [elt for elt in tmp_overall_lhs if not is_hydrogen(elt)] # Forgive H ions
        tmp_overall_rhs = [elt for elt in tmp_overall_rhs if not is_hydrogen(elt)]
        

        # Check if reactions balanced
        if is_balanced(tmp_overall_lhs, tmp_overall_rhs):
            alternates = [(tmp_overall_lhs, tmp_overall_rhs)]
        else: # Check rhea for alternates
            alternates = []
            for sma in alt_rxns[reaction_entry['ec']]:
                tmp_lhs, tmp_rhs = [[Chem.MolFromSmiles(smi) for smi in side.split(".")] for side in sma.split(">>")]
                tmp_lhs = [elt for elt in tmp_lhs if not is_hydrogen(elt)]
                tmp_rhs = [elt for elt in tmp_rhs if not is_hydrogen(elt)]
                if is_balanced(tmp_lhs, tmp_rhs):
                    alternates.append((tmp_lhs, tmp_rhs))
        
        if len(alternates) == 0:
            log.info(f"No balanced reaction found for entry {entry_id}")
            continue
        
        # Add the other direction
        rev = []
        for tmp_lhs, tmp_rhs in alternates:
            rev.append((tmp_rhs, tmp_lhs))
        alternates.extend(rev)

        for mech in reaction_entry['mechanisms']:
            if not mech['is_detailed']:
                continue

            # Assemble steps and electron flows
            elementary_steps = []
            eflows = []
            involved_in_coord_bonds = []
            misannotated_mechanism = False
            for estep in mech['steps']:
                file_path = Path(cfg.filepaths.raw_mcsa) / "mech_steps" / f"{entry_id}_{mech['mechanism_id']}_{estep['step_id']}.mrv"
                
                if not file_path.exists():
                    log.info(f"File {file_path} does not exist")
                    misannotated_mechanism = True
                    break
                
                atoms, bonds, meflows = parse_mrv(file_path)
                next_atoms, next_bonds = step(atoms, bonds, meflows)

                try:
                    lhs = construct_mols(atoms.values(), bonds.values())
                    rhs = msm(construct_mols(next_atoms.values(), next_bonds.values()))
                    
                    tmp = set()
                    for mol in lhs:
                        for atom in mol.GetAtoms():
                            if atom.GetIntProp('coord_bond') == 1:
                                tmp.add(atom.GetProp('mcsa_id'))

                    involved_in_coord_bonds.append(tmp)

                    lhs = msm(lhs)
                
                except Exception as e: # Catch errors in mechanism annotation
                    log.info(f"Error constructing mols for entry {entry_id}, mechanism {mech['mechanism_id']}, step {estep['step_id']}: {e}")
                    misannotated_mechanism = True
                    break
                
                elementary_steps.append((lhs, rhs))
                eflows.append(meflows)

            if misannotated_mechanism:
                continue

            for tmp_overall_lhs, tmp_overall_rhs in alternates:

                # Find out where reactants enter the mechanism & where products are formed in mechanism
                rct_entry_points = defaultdict(list) # Maps estep -> list[entering rcts]
                pdt_formation_points = defaultdict(list) # Maps estep -> list[forming pdts]
                
                # Backward to find rcts "latest" occurrence
                remaining_rcts = {i for i in range(len(tmp_overall_lhs))}
                for i, estep in reversed(list(enumerate(elementary_steps))):
                    if len(remaining_rcts) == 0:
                        break

                    for _, mol in enumerate(estep[0]):
                        for j, rct in enumerate(tmp_overall_lhs):
                            if j in remaining_rcts:
                                if standardize_de_atom_map(rct) == standardize_de_atom_map(mol):
                                    rct_entry_points[i].append(deepcopy(mol))
                                    remaining_rcts.remove(j)
                                    break

                # Forward to find pdts "earliest" occurrence
                remaining_pdts = {i for i in range(len(tmp_overall_rhs))}
                for i, estep in enumerate(elementary_steps):
                    if len(remaining_pdts) == 0:
                        break

                    for _, mol in enumerate(estep[1]):
                        for j, pdt in enumerate(tmp_overall_rhs):
                            if j in remaining_pdts:
                                if standardize_de_atom_map(pdt) == standardize_de_atom_map(mol):
                                    pdt_formation_points[i].append(deepcopy(mol))
                                    remaining_pdts.remove(j)
                                    break

                # Check if found all reactants and products
                overall_lhs = list(chain(*rct_entry_points.values()))
                overall_rhs = list(chain(*pdt_formation_points.values()))
                found_rcts = len(overall_lhs) >= sum(1 for m in tmp_overall_lhs)
                found_pdts = len(overall_rhs) >= sum(1 for m in tmp_overall_rhs)

                if not found_rcts or not found_pdts:
                    if not found_rcts:
                        log.info(f"Reactants not found for entry {entry_id}, mechanism {mech['mechanism_id']}")
                    if not found_pdts:
                        log.info(f"Products not found for entry {entry_id}, mechanism {mech['mechanism_id']}")
                    continue


                start2end = {}
                start2current = {}
                current2start = {}
                involved_atoms = {}
                for i, mol in enumerate(overall_lhs):
                    for atom in mol.GetAtoms():
                        pre_mech_idx = f"{i}_{atom.GetAtomMapNum()}"
                        atom.SetProp('pre_mech', pre_mech_idx)
                        start2current[pre_mech_idx] = pre_mech_idx
                        current2start[pre_mech_idx] = pre_mech_idx
                        start2end[pre_mech_idx] = None
                        involved_atoms[pre_mech_idx] = False

                for i, mol in enumerate(overall_rhs):
                    for atom in mol.GetAtoms():
                        atom.SetProp('post_mech', f"{i}_{atom.GetAtomMapNum()}")

                # Main loop
                imt_pdts = []
                for i, (estep, eflow) in enumerate(list(zip(elementary_steps, eflows))):
                    entering_rcts = rct_entry_points.get(i, [])

                    # Map start atom map numbers to end atom map numbers
                    # prior to translation
                    for pdt in pdt_formation_points.get(i, []):
                        for atom in pdt.GetAtoms():
                            start = current2start.get(atom.GetAtomMapNum(), None)
                            if start is not None:
                                start2end[start] = atom.GetProp('post_mech')

                    # Translate
                    one_step_translation, imt_rcts = translate(entering_rcts, imt_pdts, estep[0])
                    # one_step_translation = {k: v for k, v in one_step_translation.items() if k in current2start}
                    for current, _next in one_step_translation.items():
                        
                        if current in current2start:
                            start2current[current2start[current]] = _next
                        
                    current2start = {v: k for k, v in start2current.items()}

                    # Label atoms involved in this step
                    atoms_in_eflows = set(
                        chain(
                            *[chain(elt.from_, elt.to) for elt in eflow.values()]
                        )
                    ) # Collect atoms in eflows
                    atoms_involved_this_step = atoms_in_eflows.union(involved_in_coord_bonds[i]) # Add atoms involved in coordinate bonds
                    atoms_involved_this_step = [int(elt.removeprefix('a')) for elt in atoms_involved_this_step]
                    for current in current2start.keys():
                        if current in atoms_involved_this_step:
                            involved_atoms[current2start[current]] = True

                    # Transform
                    imt_pdts = transform(estep[0], estep[1], imt_rcts)

                
                if None in start2end.values():
                    log.info(f"Failed to get atom mapping for entry {entry_id}, mechanism {mech['mechanism_id']}")
                    continue

                # Translate end atom map numbers to start atom map numbers
                am = 1
                mech_atoms = [[] for _ in range(len(overall_lhs))]
                rhs_am_aidx = [{atom.GetAtomMapNum(): atom.GetIdx() for atom in mol.GetAtoms()} for mol in overall_rhs]
                for i, mol in enumerate(overall_lhs):
                    for atom in mol.GetAtoms():
                        old_am = atom.GetAtomMapNum()
                        key = f"{i}_{old_am}"
                        pdt_idx, rhs_old_am = [int(elt) for elt in start2end[key].split('_')]
                        rhs_aidx = rhs_am_aidx[pdt_idx][rhs_old_am]
                        overall_rhs[pdt_idx].GetAtomWithIdx(rhs_aidx).SetAtomMapNum(am)
                        atom.SetAtomMapNum(am)
                        mech_atoms[i].append(am)
                        am += 1

                # Append
                # Note: ignoreAtomMapNumbers option required on lhs to canonicalize SMILES
                # and indices in spite of the present use of atom map numbers to label mech atoms
                smarts = ".".join([Chem.MolToSmiles(mol, ignoreAtomMapNumbers=True) for mol in overall_lhs]) + ">>" + ".".join([Chem.MolToSmiles(mol) for mol in overall_rhs])
                if all(len(elt) == 0 for elt in mech_atoms): # This would be true if you were looking at wrong direction of reaction
                    continue
                
                mech_labeled_reactions.append([entry_id, mech['mechanism_id'], smarts, rc_to_str([mech_atoms, [[]]]), entry.get("enzyme_name"), entry.get("reference_uniprot_id"), entry["reaction"].get("ec")])

    # Save
    df = pd.DataFrame(mech_labeled_reactions, columns=columns)
    df.to_csv("mech_labeled_reactions.csv", index=False)

if __name__ == "__main__":
    main()