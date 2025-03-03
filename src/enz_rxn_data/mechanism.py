from typing import Iterable
from rdkit import Chem
from lxml import etree
from pathlib import Path
from pydantic import BaseModel
from lxml import etree

class CmlAtom(BaseModel):
    """
    Atom from CML file
    """
    id: str
    element_type: str
    x: float
    y: float
    lone_pair: int | None
    formal_charge: int
    mrv_alias: str | None
    rgroup_ref: str | None
    mrv_extra_label: str | None

class CmlBond(BaseModel):
    """
    Bond from CML file
    """
    id: str
    atom_refs: tuple[str, str]
    order: int | None
    convention: str | None = None

class CmlMEFlow(BaseModel):
    """
    MEFlow from CML file
    """
    id: str
    from_: tuple[str, str] | tuple[str]
    to: tuple[str, str] | tuple[str]

def parse_mrv(mech_step: Path) -> tuple[dict[str, CmlAtom], dict[str, CmlBond], dict[str, CmlMEFlow]]:
    """
    Parses an MRV file and extracts atoms, bonds, and MEFlow elements.
    
    Args
    ----
    mech_step:pathlib.Path
        Path to the MRV file.
    Returns
    -------
    tuple of dictionaries of atoms, bonds, and MEFlow elements.
    """
    tree = etree.parse(mech_step)
    root = tree.getroot()

    # Get namespace
    tag_ns = root.tag.split('}')
    if len(tag_ns) == 1:
        tag = ''
        ns = None
    elif len(tag_ns) == 2:
        tag = tag_ns[-1] + ':'
        ns = {tag.strip(':') : tag_ns[0].strip('{')}


    # Extract atoms
    atoms = {}
    for atom in root.xpath(f"//{tag}atom", namespaces=ns):
        atom_data = {
            'id': atom.get('id'),
            'element_type': atom.get('elementType'),
            'x': atom.get('x2'),
            'y': atom.get('y2'),
            'lone_pair': atom.get('lonePair'),
            'formal_charge': atom.get('formalCharge', 0),
            'mrv_alias': atom.get('mrvAlias'),
            'rgroup_ref': atom.get('rgroupRef'),
            'mrv_extra_label': atom.get('mrvExtraLabel')
        }
        catom = CmlAtom(**atom_data)
        atoms[catom.id] = catom

    # Extract bonds
    bonds = {}
    for bond in root.xpath(f"//{tag}bond", namespaces=ns):
        bond_data = {
            'id': bond.get('id'),
            'atom_refs': tuple(bond.get('atomRefs2').split(' ')),
            'order': bond.get('order'),
            'convention': bond.get('convention')
        }
        cbond = CmlBond(**bond_data)
        bonds[cbond.atom_refs] = cbond

    # Extract MEFlow elements
    meflows = {}
    for meflow in root.xpath(f"//{tag}MEFlow", namespaces=ns):
        from_to = [child.get('atomRef') or child.get('atomRefs') for child in meflow]
        from_to = [elt.replace('m1.', '') for elt in from_to if elt is not None]
        meflow_data = {
            'id': meflow.get('id'),
            'from_': tuple(from_to[0].split(' ')) if len(from_to) > 0 else None,
            'to': tuple(from_to[1].split(' ')) if len(from_to) > 1 else None,
        }
        cmeflow = CmlMEFlow(**meflow_data)
        meflows[cmeflow.id] = cmeflow

    return atoms, bonds, meflows

def construct_mols(cml_atoms: Iterable[CmlAtom], cml_bonds: Iterable[CmlBond]) -> Iterable[Chem.Mol]:
    '''
    Constructs RDKit molecules from CML atoms and bonds.
    '''
    rw_mol = Chem.RWMol()
    mcsa2rdkit = {}
    bond_types = {
        1: Chem.rdchem.BondType.SINGLE,
        2: Chem.rdchem.BondType.DOUBLE,
        3: Chem.rdchem.BondType.TRIPLE
    }
    for catom in cml_atoms:
        if catom.element_type == 'R':
            aidx = rw_mol.AddAtom(Chem.Atom('*'))
        else:
            aidx = rw_mol.AddAtom(Chem.Atom(catom.element_type))
        
        mcsa2rdkit[catom.id] = aidx
        rw_mol.GetAtomWithIdx(aidx).SetProp('mcsa_id', catom.id)
        rw_mol.GetAtomWithIdx(aidx).SetAtomMapNum(int(catom.id.strip('a')))
        rw_mol.GetAtomWithIdx(aidx).SetFormalCharge(catom.formal_charge)
        
        if catom.mrv_alias is not None:
            rw_mol.GetAtomWithIdx(aidx).SetProp('mrv_alias', catom.mrv_alias)
        
        if catom.rgroup_ref is not None:
            rw_mol.GetAtomWithIdx(aidx).SetProp('rgroup_ref', catom.rgroup_ref)

        if catom.mrv_extra_label is not None:
            rw_mol.GetAtomWithIdx(aidx).SetProp('mrv_extra_label', catom.mrv_extra_label)

    for cbond in cml_bonds:
        bond_type = bond_types.get(cbond.order)

        if bond_type is None:
            print(f"Ignoring orderless bond: {cbond.id} of convention {cbond.convention}")
            continue
        
        from_, to = [mcsa2rdkit[atom_ref] for atom_ref in cbond.atom_refs]

        rw_mol.AddBond(from_, to, order=bond_type)

    Chem.SanitizeMol(rw_mol)
    mols = Chem.rdmolops.GetMolFrags(rw_mol, asMols=True)

    return mols

def get_overall_reaction(compounds: Iterable[dict[str, str | int]], mol_path: Path) -> tuple[tuple[Chem.Mol], tuple[Chem.Mol]]:
    '''
    Constructs the overall reaction from the given compounds.
    
    Args
    ----
    compounds: Iterable[dict[str, str | int]]
        List of compounds with their Chebi ID and type (reactant or product).
    mol_path: Path
        Path to the directory containing the mol files.
    
    Returns
    -------
    tuple[tuple[Chem.Mol], tuple[Chem.Mol]]
        Tuple of tuples containing the reactants and products as RDKit molecules.
    '''
    lhs = []
    rhs = []
    for c in compounds:
        for _ in range(c['count']):
            mol = Chem.MolFromMolFile(mol_path / f"{c['chebi_id']}.mol")
            Chem.RemoveStereochemistry(mol)

            if c['type'] == 'reactant':
                lhs.append(mol)
            elif c['type'] == 'product':
                rhs.append(mol)

    return tuple(lhs), tuple(rhs)

def step(cml_atoms: dict[str, CmlAtom], cml_bonds: dict[str, CmlBond], cml_meflows: dict[str, CmlMEFlow]) -> tuple[dict[str, CmlAtom], dict[str, CmlBond]]:
    """
    Updates the bonds in the CML file based on the MEFlow elements.
    
    Args
    ----
    cml_atoms: dict[str, CmlAtom]
        Dictionary of CML atoms.
    cml_bonds: dict[str, CmlBond]
        Dictionary of CML bonds.
    cml_meflows: dict[str, CmlMEFlow]
        Dictionary of CML MEFlow elements.
    
    Returns
    -------
    Updated atoms and bonds
    """
    cml_atoms = {k: v.model_copy(deep=True) for k, v in cml_atoms.items()} # Defensive copy
    cml_bonds = {k: v.model_copy(deep=True) for k, v in cml_bonds.items()} # Defensive copy
    for meflow in cml_meflows.values():
        if len(meflow.from_) == 2 and len(meflow.to) == 2:
            cml_bonds[meflow.from_].order -= 1
            if cml_bonds.get(meflow.to):
                cml_bonds[meflow.to].order += 1
            else:
                cml_bonds[meflow.to] = CmlBond(id='added', atom_refs=meflow.to, order=1)
        elif len(meflow.from_) == 2 and len(meflow.to) == 1:
            cml_atoms[meflow.to[0]].formal_charge -= 1
            cml_bonds[meflow.from_].order -= 1
        elif len(meflow.from_) == 1 and len(meflow.to) == 2:
            cml_atoms[meflow.from_[0]].formal_charge += 1
            if cml_bonds.get(meflow.to):
                cml_bonds[meflow.to].order += 1
            else:
                cml_bonds[meflow.to] = CmlBond(id='added', atom_refs=meflow.to, order=1)

    return cml_atoms, {k: v for k, v in cml_bonds.items() if v.order is not None and v.order > 0}

if __name__ == "__main__":
    import json
    for i in range(1, 7):
        file_path = f'/home/stef/enz_rxn_data/data/raw/mcsa/mech_steps/219_2_{i}.mrv'
        atoms, bonds, meflows = parse_mrv(file_path)
        mols = construct_mols(atoms.values(), bonds.values())
        next_atoms, next_bonds = step(atoms, bonds, meflows)
        next_mols = construct_mols(next_atoms.values(), next_bonds.values())
        print([Chem.MolToSmiles(mol) for mol in mols])
        print([Chem.MolToSmiles(mol) for mol in next_mols])

        # mcsa_path = Path("/home/stef/enz_rxn_data/data/raw/mcsa")
        # mol_path = mcsa_path / "mols"
        # with open(mcsa_path / "entries_7.json", "r") as f:
        #     entries = json.load(f)

        # cpds = entries['722']['reaction']['compounds']
        # lhs, rhs = get_overall_reaction(cpds, mol_path)
        # print([Chem.MolToSmiles(mol) for mol in lhs])
        # print([Chem.MolToSmiles(mol) for mol in rhs])