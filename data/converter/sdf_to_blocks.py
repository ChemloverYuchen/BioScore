#!/usr/bin/python
# -*- coding:utf-8 -*-

from typing import List, Optional

from data.dataset import Block, Atom

from .atom_blocks_to_frag_blocks import atom_blocks_to_frag_blocks

from rdkit import Chem

from rdkit.Chem import SDMolSupplier
from typing import List, Optional

def sdf_to_blocks(sdf_file: str, using_hydrogen: bool = False, molecule_type: Optional[str] = "small", fragment: bool=False) -> List[Block]:
    '''
    Convert an SDF file to a list of lists of blocks for each molecule.
    
    Parameters:
        sdf_file: Path to the SDF file
        using_hydrogen: Whether to preserve hydrogen atoms, default false
        molecule_type:  "small" (small molecule). If not specified, deduce from the SDF file

    Returns:
        A list of blocks representing a small molecule, etc.
    '''
    # Read SDF file using RDKit
    supplier = SDMolSupplier(sdf_file)
    mols = [mol for mol in supplier if mol is not None]
    
    # Ensure that at least one molecule is found
    assert len(mols) > 0, "No valid molecules found in the SDF file."
    assert len(mols) == 1, "sdf conteins molecules more than 1"
    blocks = []
    mol = mols[0]
    # Iterate over each molecule in the SDF file
    
    # If molecule type is 'small', treat it as a small molecule
    if molecule_type == 'small':
        remap = {}
        for i, atom in enumerate(mol.GetAtoms()):
            # atom_index = int(atom.GetIdx())
            atom_name = atom.GetSymbol()
            atom_coords = mol.GetConformer().GetAtomPosition(i)
            atom_element = atom.GetSymbol()
#             print(atom_name, atom_element)
            if not using_hydrogen and atom_element == 'H':
                continue

            # Create Atom object and add to blocks
            atom_obj = Atom(atom_name, [atom_coords.x, atom_coords.y, atom_coords.z], atom_element)
            # Create a block for each atom
            # blocks.append(Block(atom_element.lower(), [atom_obj], atom))
            blocks.append(Block(atom_element.lower(), [atom_obj]))
            remap[i + 1] = len(remap) # atom indexes in the records start from 1
        if fragment:
            bonds = []
            for bond in mol.GetBonds():
                src_idx = bond.GetBeginAtomIdx()
                dst_idx = bond.GetEndAtomIdx()
                bond_type = bond.GetBondTypeAsDouble()

                if src_idx not in remap or dst_idx not in remap:
                    continue

                bonds.append((remap[src_idx], remap[dst_idx], bond_type))
#             print(bonds, remap)
            blocks = atom_blocks_to_frag_blocks(blocks, bonds=bonds)

    elif molecule_type == 'protein':
        raise NotImplementedError(f'Molecule type {molecule_type} not implemented')

    else:
        raise NotImplementedError(f'Molecule type {molecule_type} not implemented')

    # Return the final list of blocks
    assert len(blocks) != 0, 'list_blocks is [] !'
    return blocks


if __name__ == '__main__':
    import sys
    list_blocks = sdf_to_blocks(sys.argv[1])
    print(f'{sys.argv[1]} parsed')
    print(f'number of blocks: {len(list_blocks)}')