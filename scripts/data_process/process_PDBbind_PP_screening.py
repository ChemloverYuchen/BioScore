#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import sys
import json
import math
import pickle
import argparse

import pandas as pd
from joblib import Parallel, delayed  # for multiprocessing

PROJ_DIR = os.path.join(
    os.path.split(os.path.abspath(__file__))[0],
    '..', '..'
)
sys.path.append(PROJ_DIR)

from utils.logger import print_log
from utils.convert import kd_to_dg
from utils.network import fetch_from_pdb
from data.pdb_utils import Complex, Protein, Residue, VOCAB


def parse():
    parser = argparse.ArgumentParser(description='Process protein-protein binding affinity data from the structural benchmark')
    parser.add_argument('--pdb_dir', type=str, required=True,
                        help='Directory of all pdb dirs, each pdb dir: 79 * (1 + 78) * 100 decoys + info')
    parser.add_argument('--start', type=int, required=True,
                        help='start pdb_id idx')
    parser.add_argument('--end', type=int, required=True,
                        help='end pdb_id idx')
    parser.add_argument('--out_dir', type=str, required=True,
                        help='Output directory')
    parser.add_argument('--outname', type=str, required=True,
                        help='Output name.pkl')
    parser.add_argument('--interface_dist_th', type=float, default=6.0,
                        help='Residues who has atoms with distance below this threshold are considered in the complex interface')
    return parser.parse_args()


def residue_to_pd_rows(chain: str, residue: Residue):
    rows = []
    res_id, insertion_code = residue.get_id()
    resname = residue.real_abrv if hasattr(residue, 'real_abrv') else VOCAB.symbol_to_abrv(residue.get_symbol())
    for atom_name in residue.get_atom_names():
        atom = residue.get_atom(atom_name)
        if atom.element == 'H':  # skip hydrogen
            continue
        rows.append((
            chain, insertion_code, res_id, resname,
            atom.coordinate[0], atom.coordinate[1], atom.coordinate[2],
            atom.element, atom.name
        ))
    return rows


def process_line(line, root, pdb_id, interface_dist_th):
    line = line.split(',')
    assert len(line) == 4
    struct_idx, (rec_chains, lig_chains) = line[0], line[2].strip().split("_")  # e.g. 1emv-1emv_decoy_0,1emv,A_B,Yes
    is_native = line[3]
    
    lig_chains, rec_chains = list(lig_chains), list(rec_chains)
    item = {
            'id': f'{struct_idx}',
            'native': is_native
    }
    item['affinity'] = {
            'Kd': 0.0,
            'dG': 0.0,
            'neglog_aff': 0.0,
    }
    
    cplx_id, _, decoy_id = struct_idx.split('_')
    cplx_path = os.path.join(root, pdb_id, cplx_id, f'decoy_{decoy_id}.pdb')
    
    # rank by seq lengths [longer, shorter]
    prot = Protein.from_pdb(cplx_path)
    peptides = prot.peptides
    seq_protein1 = ''.join([peptides[c].get_seq() for c in rec_chains])
    seq_protein2 = ''.join([peptides[c].get_seq() for c in lig_chains])
    if len(seq_protein1) >= len(seq_protein2):
        pass
    else:
        shorter_chains = rec_chains
        rec_chains, lig_chains = lig_chains, shorter_chains

    cplx = Complex(item['id'], peptides, rec_chains, lig_chains)

    
    # Protein1 is receptor, protein2 is ligand（rank by seq lengths [longer, shorter]）
    item['seq_protein1'] = ''.join([cplx.get_chain(c).get_seq() for c in rec_chains])
    item['chains_protein1'] = rec_chains
    item['seq_protein2'] = ''.join([cplx.get_chain(c).get_seq() for c in lig_chains])
    item['chains_protein2'] = lig_chains

    # construct pockets
    interface1, interface2, rec_index, lig_index, rec_seqs, lig_seqs  = cplx.get_interacting_residues(dist_th=interface_dist_th)
    if len(interface1) == 0:  # no interface (if len(interface1) == 0 then we must have len(interface2) == 0)
        print_log(f'{pdb} has no interface', level='ERROR')
        return None
    columns = ['chain', 'insertion_code', 'residue', 'resname', 'x', 'y', 'z', 'element', 'name']
    for i, interface in enumerate([interface1, interface2]):
        data = []
        for chain, residue in interface:
            data.extend(residue_to_pd_rows(chain, residue))
        item[f'atoms_interface{i + 1}'] = pd.DataFrame(data, columns=columns)
            
    # construct DataFrame of coordinates
    for i, chains in enumerate([rec_chains, lig_chains]):
        data = []
        for chain in chains:
            chain_obj = cplx.get_chain(chain)
            if chain_obj is None:
                print_log(f'{chain} not in {pdb}: {cplx.get_chain_names()}. Skip this chain.', level='WARN')
                continue
            for residue in chain_obj:
                data.extend(residue_to_pd_rows(chain, residue))                
        item[f'atoms_protein{i + 1}'] = pd.DataFrame(data, columns=columns)

    item["rec_seqs"] = rec_seqs
    item["lig_seqs"] = lig_seqs
    assert len(rec_seqs) >= len(lig_seqs)
    
    return item


def process_single_line(line, root, pdb_id, interface_dist_th):
    """ process single line, return processed data """
    item = process_line(line, root, pdb_id, interface_dist_th)
    if item in ('', None):  # filter invalid
        return None
    return item
    

def main(args):
    root = args.pdb_dir
    pdb_id_ls = sorted(os.listdir(root))
    
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    
    total = 0
    for pdb_id in pdb_id_ls[args.start:args.end]:
        index_file = os.path.join(root, pdb_id, f'{pdb_id}_info.csv')

        with open(index_file, 'r') as fin:
            lines = fin.readlines()
        lines = lines[1:]  # the first one is head
        
        # multi-process
        print_log(f'Preprocessing {pdb_id}')
        n_jobs = 60
        processed_pdbbind = Parallel(n_jobs=n_jobs)(
            delayed(process_single_line)(line, root, pdb_id, args.interface_dist_th)
            for line in lines
        )
        
        # filter
        processed_pdbbind = [item for item in processed_pdbbind if item is not None]
        
        # summary
        print_log(f'{pdb_id}: valid/total = {len(processed_pdbbind)}/{len(lines)}')
        
        # save to file
        database_out = os.path.join(args.out_dir, f'{args.outname}_{pdb_id}.pkl')
        total += len(processed_pdbbind)
        
        print_log(f'Obtained {len(processed_pdbbind)} data after filtering, saving to {database_out}...')
        with open(database_out, 'wb') as fout:
            pickle.dump(processed_pdbbind, fout)
        
    print_log('Binary file saved.')
    print_log(f'Finished! Total: {total}')

if __name__ == '__main__':
    main(parse())