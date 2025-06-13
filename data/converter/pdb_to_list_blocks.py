#!/usr/bin/python
# -*- coding:utf-8 -*-
from typing import List, Optional

import MDAnalysis as mda

from data.dataset import Block, Atom, VOCAB
def pdb_to_list_blocks(pdb: str, selected_chains: Optional[List[str]]=None) -> List[List[Block]]:
    '''
        Convert pdb file to a list of lists of blocks.
        Each chain will be a list of blocks.
        
        Parameters:
            pdb: Path to the pdb file
            selected_chains: List of selected chain ids. The returned list will be ordered
                according to the ordering of chain ids in this parameter. If not specified,
                all chains will be returned. e.g. ['A', 'B']

        Returns:
            A list of lists of blocks. Each chain in the pdb file will be parsed into
            one list of blocks.
            example:
                [
                    [residueA1, residueA2, ...],  # chain A
                    [residueB1, residueB2, ...]   # chain B
                ],
                where each residue is instantiated by Block data class.
    '''
    u = mda.Universe(pdb)
    list_blocks, chain_ids = [], {}

    res_ids, residues = {}, []
    last_chain_id = None

    processed_chain_id = []
    for residue in u.residues:
        # pass water and metal ions
        abrv = residue.resname
        if abrv in ['WAT', 'HOH']: 
            continue
        if abrv in ['NA', 'K', 'CA', 'MG', 'ZN', 'FE', 'CU', 'MN', 'CO', 'NI']: 
            continue
        
        chain_id = residue.segid

        if last_chain_id and chain_id != last_chain_id:  # new chain, save the last chain, clear residues
            processed_chain_id.append(last_chain_id)
            # the last few residues might be non-relevant molecules in the solvent if their types are unk
            end = len(residues) - 1
            while end >= 0:
                if residues[end].symbol == VOCAB.UNK:
                    end -= 1
                else:
                    break
            residues = residues[:end + 1]
            if len(residues) == 0:  # not a chain
                residues = []
                continue
            
            chain_ids[last_chain_id] = len(list_blocks)
            list_blocks.append(residues)
            residues = []

        # main
        if selected_chains is not None and chain_id not in selected_chains:  
            continue
        abrv = residue.resname
        # res_idx = residue.ix
        res_number = residue.resid
        insert_code = residue.icode
        res_id = f'{chain_id}-{res_number}-{insert_code}'
            
        if res_id in res_ids:
            continue
        
        if abrv == 'MSE':
            abrv = 'MET'

        symbol = VOCAB.abrv_to_symbol(abrv)
        atoms = [ Atom(atom.name, atom.position, atom.element) for atom in residue.atoms if atom.element != 'H' ]
        if len(atoms) == 0:
            continue
        residues.append(Block(symbol, atoms)) 
        
        res_ids[res_id] = True

        last_chain_id = chain_id
    
    # the last few residues might be non-relevant molecules in the solvent if their types are unk
    end = len(residues) - 1
    while end >= 0:
        if residues[end].symbol == VOCAB.UNK:
            end -= 1
        else:
            break
    residues = residues[:end + 1]
    if len(residues) > 0:  # not a chain
        chain_ids[last_chain_id] = len(list_blocks)
        list_blocks.append(residues)
 
    # reorder
    if selected_chains is not None:
        list_blocks = [list_blocks[chain_ids[chain_id]] for chain_id in selected_chains]
    
    assert len(list_blocks) != 0, 'list_blocks is [] !'
    return list_blocks

if __name__ == '__main__':
    import sys
    list_blocks = pdb_to_list_blocks(sys.argv[1])
    print(f'{sys.argv[1]} parsed')
    print(f'number of chains: {len(list_blocks)}')
    for i, chain in enumerate(list_blocks):
        print(f'chain {i} lengths: {len(chain)}')