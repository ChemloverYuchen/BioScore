#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import sys
import json
import pickle
import argparse

import numpy as np
from joblib import Parallel, delayed  # for multiprocessing

PROJ_DIR = os.path.join(
    os.path.split(os.path.abspath(__file__))[0],
    '..', '..'
)
print(f'Project directory: {PROJ_DIR}')
sys.path.append(PROJ_DIR)

from utils.logger import print_log
from data.pdb_utils import Residue, VOCAB
from data.dataset import blocks_interface, blocks_to_data
from data.converter.pdb_to_list_blocks import pdb_to_list_blocks
from data.converter.mol2_to_blocks import mol2_to_blocks
from data.converter.sdf_to_blocks import sdf_to_blocks

def parse():
    parser = argparse.ArgumentParser(description='Process PDBbind benchmark of protein-ligand interaction')
    parser.add_argument('--benchmark_dir', type=str, required=True,
                        help='Directory of the benchmark containing metadata and pdb_files')
    parser.add_argument('--out_dir', type=str, required=True,
                        help='Output directory')
    parser.add_argument('--json_name', type=str, default='affinities_test',
                        help='Affinity json file name')
    parser.add_argument('--outname', type=str, default='PDBbind_PL_test',
                        help='Output file name')
    parser.add_argument('--fragment', default=None, choices=['PS_300', 'PS_500'], help='Use fragment-based representation of small molecules')
    parser.add_argument('--interface_dist_th', type=float, default=10.0,
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


def process_one(pdb_id, label, benchmark_dir, interface_dist_th, fragment):

    item = {}
    item['id'] = pdb_id  # pdb code, e.g. 1fc2
    item['affinity'] = { 'neglog_aff': label }
    pdb_dir = os.path.join(benchmark_dir, 'processed_data')

    prot_fname = os.path.join(pdb_dir, pdb_id, f'{pdb_id}_protein_pocket_{interface_dist_th}.pdb')
    sm_fname = os.path.join(pdb_dir, pdb_id, f'{pdb_id}_ligand.sdf')

    try:
        list_blocks1 = pdb_to_list_blocks(prot_fname)
    except Exception as e:
        print_log(f'{pdb_id} protein parsing failed: {e}', level='ERROR')
        return None


    try:
        blocks2 = sdf_to_blocks(sm_fname, fragment=fragment)
    except Exception as e:
        print_log(f'{pdb_id} ligand parsing failed: {e}', level='ERROR')
        return None


    blocks1 = []
    for b in list_blocks1:
        blocks1.extend(b)

    data = blocks_to_data(blocks1, blocks2)
    for key in data:
        if isinstance(data[key], np.ndarray):
            data[key] = data[key].tolist()
    
    item['data'] = data
   
    return item


def process_single_line(pdb_id, label, benchmark_dir, interface_dist_th, use_fragment):
    """ process single line, return processed data  """
    print_log(f'PDB_ID: {pdb_id}')
    item = process_one(pdb_id, label, benchmark_dir, interface_dist_th, use_fragment)
    if item in ('', None):  # drop invalid
        return None
    return item


def main(args):

    # preprocess PDBbind into json summaries and complex pdbs
    labels = json.load(open(os.path.join(args.benchmark_dir, 'metadata', f'{args.json_name}.json'), 'r'))
    label_ls = [k for k in labels]
    print_log('Preprocessing')
    processed_pdbbind = []
    cnt = 0

    # multi-process
    n_jobs = 32
    processed_pdbbind = Parallel(n_jobs=n_jobs)(
        delayed(process_single_line)(pdb_id, labels[pdb_id], args.benchmark_dir, args.interface_dist_th, args.fragment is not None)
        for pdb_id in label_ls
    )

    # filter
    processed_pdbbind = [item for item in processed_pdbbind if item is not None]
    
    # summary
    print_log(f'valid/total = {len(processed_pdbbind)}/{len(label_ls)}')
        
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    
    data_out_path = os.path.join(args.out_dir, f'{args.outname}.pkl')
    print_log(f'Obtained {len(processed_pdbbind)}, saving to {data_out_path}...')
    with open(data_out_path, 'wb') as fout:
        pickle.dump(processed_pdbbind, fout)

    print_log('Finished!')


if __name__ == '__main__':
    main(parse())
