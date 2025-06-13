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
    parser.add_argument('--outname', type=str, default='CASF2016_docking',
                        help='Output file name')
    
    parser.add_argument('--fragment', default=None, choices=['PS_300', 'PS_500'], help='Use fragment-based representation of small molecules')
    parser.add_argument('--interface_dist_th', type=float, default=10.0,
                        help='Residues who has atoms with distance below this threshold are considered in the complex interface')
    return parser.parse_args()


def process_line(pdb_id, line, struct_dir, interface_dist_th, fragment):
    line = line.strip().split(',')  # e.g. 1a30_128,7.02
    assert len(line) == 2
    
    struct_idx, rmsd = line[0], float(line[1])
    assert struct_idx.split('_')[0] == pdb_id
    
    item = {
        'id': struct_idx,
        'rmsd': rmsd
    }
    item['affinity'] = { 'neglog_aff': 0.0 }
    
    prot_fname = os.path.join(struct_dir, f'{pdb_id}_protein_pocket_{interface_dist_th}.pdb')
    sm_fname = os.path.join(struct_dir, f'{struct_idx}.sdf')

    try:
        list_blocks1 = pdb_to_list_blocks(prot_fname)
    except Exception as e:
        print_log(f'{pdb_id} protein parsing failed: {e}', level='ERROR')
        return None
    try:
        blocks2 = sdf_to_blocks(sm_fname, fragment=fragment)
    except Exception as e:
        print_log(f'{struct_idx} parsing failed: {e}', level='ERROR')
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


def process_single_line(pdb_id, line, struct_dir, interface_dist_th, use_fragment):
    """ process single line, return processed data  """
    item = process_line(pdb_id, line, struct_dir, interface_dist_th, use_fragment)
    if item in ('', None):  # drop invalid
        return None
    return item


def main(args):

    # get pdb_ids
    pdb_id_ls = sorted([pdb_id for pdb_id in os.listdir(args.benchmark_dir) if len(pdb_id) == 4])
    print(len(pdb_id_ls))
    
    processed_pdbbind = []
    cnt = 0
    
    # load index file and preprocess
    for pdb_id in pdb_id_ls:
        print_log(f'Preprocessing {pdb_id} ...')
        struct_dir = os.path.join(args.benchmark_dir, pdb_id)
        
        index_file = os.path.join(struct_dir, f'{pdb_id}_rmsd.csv')
        with open(index_file, 'r') as f:
            lines = f.readlines()[1:] # skip headline
            
        # multi-process
        n_jobs = 60
        processed_pdbbind_single = Parallel(n_jobs=n_jobs)(
            delayed(process_single_line)(pdb_id, line, struct_dir, args.interface_dist_th, args.fragment is not None)
            for line in lines
        )
        cnt += len(processed_pdbbind_single)
        
        # filter
        processed_pdbbind_single = [item for item in processed_pdbbind_single if item is not None]
        
        # combine
        processed_pdbbind.extend(processed_pdbbind_single)
        
        # summary
        print_log(f'valid/total = {len(processed_pdbbind)}/{cnt}')
        

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
        
    data_out_path = os.path.join(args.out_dir, f'{args.outname}.pkl')
    print_log(f'Obtained {len(processed_pdbbind)}, saving to {data_out_path}...')
    with open(data_out_path, 'wb') as fout:
        pickle.dump(processed_pdbbind, fout)

    print_log('Finished!')


if __name__ == '__main__':
    main(parse())
