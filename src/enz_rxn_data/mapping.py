from rdkit import Chem
import numpy as np
import pandas as pd
from collections import Counter
from ergochemics.mapping import extract_operator_patts

def get_bond_matrix(side: list[Chem.Mol], n_atoms: int) -> tuple[np.ndarray, dict[int, str]]:
    '''
    Get the bond matrix for a list of molecules

    Args
    ----
    side: list[Chem.Mol]
        A list of molecules
    n_atoms: int
        The total number of atoms in the molecules
    
    Returns
    -------
    A: np.ndarray
        The bond matrix
    element_symbols: dict[int, str]
        A dictionary of atom indices and their corresponding element symbols
    '''
    A = np.zeros(shape=(n_atoms, n_atoms), dtype=np.float32)
    element_symbols = {}
    for mol in side:
        for bond in mol.GetBonds():
            ai, aj = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            i = mol.GetAtomWithIdx(ai).GetAtomMapNum() - 1
            j = mol.GetAtomWithIdx(aj).GetAtomMapNum() - 1
            A[i, j] = A[j, i] = bond.GetBondTypeAsDouble()

        for atom in mol.GetAtoms():
            i = atom.GetAtomMapNum() - 1
            elt_i = atom.GetSymbol()
            
            if i not in element_symbols:
                element_symbols[i] = elt_i


    return A, element_symbols

def count_bond_changes(rxn: str) -> Counter[tuple[str, float, float], int]:
    """
    Count bond changes of a certain type defined by the atomic
    constituents and the bond order on each side of the reaction

    Args
    ----
    rxn: str
        The atom-mapped reaction
    
    Returns
    -------
    
    """
    bond_changes = []
    lhs, rhs = [[Chem.MolFromSmiles(s) for s in side.split(".")] for side in rxn.split(">>")]
    nL = sum(m.GetNumAtoms() for m in lhs)
    nR = sum(m.GetNumAtoms() for m in rhs)
    
    if nL != nR:
        raise ValueError("Number of atoms in reactants and products do not match")
    
    L, element_symbols = get_bond_matrix(lhs, nL)
    R, _ = get_bond_matrix(rhs, nR)
    D = R - L
    nz = np.vstack(np.where(D != 0)).T
    for i, j in nz:
        i, j = int(i), int(j)
        atomic_constituents = "".join(sorted(element_symbols[i] + element_symbols[j]))
        bond_orders = [float(L[i, j]), float(R[i, j])]
        bond_changes.append(tuple([atomic_constituents] + bond_orders))

    bond_counts = Counter(bond_changes)
    
    return bond_counts

def does_break_cc(rxn: str) -> bool:
    """
    Check if the reaction breaks a C-C bond

    Args
    ----
    rxn: str
        The atom-mapped reaction
    
    Returns
    -------
    bool
        True if the reaction breaks a C-C bond, False otherwise
    """
    bond_cts = count_bond_changes(rxn)
    return bond_cts[("CC", 1.0, 0.0)] > 0

def has_intramol_breaks(rule: str) -> bool:
    '''
   Check if rule has '.' notation that leaves
   out parts of a molecule.
    '''
    lhs, rhs = extract_operator_patts(rule)

    for elt in lhs:
        if "." in elt:
            return True
    for elt in rhs:
        if "." in elt:
            return True
    return False

'''
Multiple mapping resolvers
'''

def min_rule_resolver(group: pd.DataFrame, rule_cts: dict) -> pd.Series:
    '''
    First looks for rules w/o intramolecular breaks, then looks for 
    rules that don't break C-C bonds, then chooses the most common rule.

    Args
    ----
    group: pd.DataFrame
        A group of mappings for a single unique reaction. Must contain
        columns: "am_smarts", "rule"
    rule_cts: dict
        A dictionary mapping rules to their counts in the full dataset
    Returns
    -------
    pd.Series
        The selected mapping from the group
    '''
    if len(group) == 1:
        return group.iloc[0]
    else:
        intramol_breaks = group["rule"].apply(has_intramol_breaks)

        # If a subset are specific, use those, proceed to cc break check
        # otherwise, resort to those with intramolecular breaks
        if intramol_breaks.any() and not intramol_breaks.all():
            group = group.loc[~intramol_breaks]

        cc_breaks = group["am_smarts"].apply(does_break_cc)

        if cc_breaks.all() or not cc_breaks.any(): # If all or none break cc, doesn't matter, use counts
            return group.loc[group["rule"].map(rule_cts).idxmax()]
        else:
            not_cc = group.loc[~cc_breaks] # If a subset break cc, use those that don't
            return not_cc.loc[not_cc["rule"].map(rule_cts).idxmax()]
        
def largest_subgraph(group: pd.DataFrame, rule_cts: dict = {}) -> pd.Series:
    '''
    Chooses among multiple rules mapped to a reaction that which has
    the largest subgraph, i.e. the most atoms in the SMARTS pattern

    Args
    ----
    group: pd.DataFrame
        A group of mappings for a single unique reaction. Must contain
        columns: "template_aidxs" referring to the atom indices of all atoms
        matched to the rule SMARTS template.
    rule_cts: dict (For compatibility)
        A dictionary mapping rules to their counts in the full dataset
    Returns
    -------
    pd.Series
        The selected mapping from the group
    '''
    if len(group) == 1:
        return group.iloc[0]
    
    subgraph_sizes = group["template_aidxs"].apply(lambda x: sum(len(elt) for elt in x[0]))
    return group.loc[subgraph_sizes.idxmax()]