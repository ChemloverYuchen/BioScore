#!/usr/bin/python
# -*- coding:utf-8 -*-
import re
import os
import sys
import math
import pickle
import argparse
from argparse import Namespace

import torch

import pandas as pd
from joblib import Parallel, delayed  # for multiprocessing

PROJ_DIR = os.path.join(
    os.path.split(os.path.abspath(__file__))[0],
    '..', '..'
)
print(f'Project directory: {PROJ_DIR}')
sys.path.append(PROJ_DIR)

from utils.convert import kd_to_dg
from utils.network import url_get
from utils.logger import print_log
from data.pdb_utils import Complex, Residue, VOCAB, Protein, Peptide


def parse():
    parser = argparse.ArgumentParser(description='Process PDBbind benchmark of protein-protein interaction')
    parser.add_argument('--index_file', type=str, required=True,
                        help='Path to the index file')
    parser.add_argument('--pdb_dir', type=str, required=True,
                        help='Directory of pdbs')
    parser.add_argument('--out_dir', type=str, required=True,
                        help='Output directory')
    parser.add_argument('--outname', type=str, default='PDBbind_PP_test',
                        help='Output file name')
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


def process_line(line, pdb_dir, interface_dist_th):
    item = {}
    
    line = line.strip().split(',')  # e.g. 2uyz,A_B,7.086186147616282
    struct_idx, (rec_chains, lig_chains) = line[0], line[1].split('_')
    affinity = float(line[2])
    pdb_data_path = os.path.join(pdb_dir, f'{struct_idx}.pdb')

    item['id'] = struct_idx  # struct_idx
    
    # affinity data
    item['affinity'] = {
        'Kd': 0.0,
        'dG': 0.0,
        'neglog_aff': affinity
    }
    
    # structure data
    rec_chains, lig_chains = list(rec_chains), list(lig_chains)
    cplx_path = pdb_data_path
    
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
    
    # Protein1 is receptor, protein2 is ligand （rank by seq lengths [longer, shorter]）
    item['seq_protein1'] = ''.join([cplx.get_chain(c).get_seq() for c in rec_chains])
    item['chains_protein1'] = rec_chains
    item['seq_protein2'] = ''.join([cplx.get_chain(c).get_seq() for c in lig_chains])
    item['chains_protein2'] = lig_chains

    # construct pockets
    interface1, interface2, rec_index, lig_index, rec_seqs, lig_seqs = cplx.get_interacting_residues(dist_th=interface_dist_th)
    print(f'interface: {len(interface1)}, {len(interface2)}')
    if len(interface1) == 0:  # no interface (if len(interface1) == 0 then we must have len(interface2) == 0)
        print_log(f'{struct_idx} has no interface', level='ERROR')
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
                print_log(f'{chain} not in {struct_idx}: {cplx.get_chain_names()}. Skip this chain.', level='WARN')
                continue
            for residue in chain_obj:
                data.extend(residue_to_pd_rows(chain, residue))                
        item[f'atoms_protein{i + 1}'] = pd.DataFrame(data, columns=columns)
        
    item["rec_seqs"] = rec_seqs
    item["lig_seqs"] = lig_seqs

    return item


def process_single_line(line, pdb_dir, interface_dist_th):
    """ process single line, return processed data  """
    try:
        item = process_line(line, pdb_dir, interface_dist_th)
    except Exception as e:
        item = None
        struct_idx = line.split(',')[0]
        print_log(f'{struct_idx} parsing failed: {e}', level='ERROR')
    if item in ('', None):  # drop invalid
        return None
    return item


def main(args):

    # TODO: 1. preprocess PDBbind into json summaries and complex pdbs
    print_log('Preprocessing ...')
    with open(args.index_file, 'r') as fin:
        lines = fin.readlines()[1:]
        
    processed_pdbbind = []
    cnt = 0
    
    # multi-process
    n_jobs = 32
    processed_pdbbind = Parallel(n_jobs=n_jobs)(
        delayed(process_single_line)(line, args.pdb_dir, args.interface_dist_th)
        for line in lines
    )
    cnt += len(processed_pdbbind)
    
    # filter
    processed_pdbbind = [item for item in processed_pdbbind if item is not None]
    
    # summary
    print_log(f'valid/total = {len(processed_pdbbind)}/{len(lines)}')

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
        
    database_out = os.path.join(args.out_dir, f'{args.outname}.pkl')
    print_log(f'Obtained {len(processed_pdbbind)} data after filtering, saving to {database_out}...')
    with open(database_out, 'wb') as fout:
        pickle.dump(processed_pdbbind, fout)
        
    print_log('Finished!')


if __name__ == '__main__':
    main(parse())
