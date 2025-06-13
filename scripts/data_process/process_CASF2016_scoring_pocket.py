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
    parser.add_argument('--index_file', type=str, required=True,
                        help='Index file for the benchmark')
    parser.add_argument('--benchmark_dir', type=str, required=True,
                        help='Directory of the benchmark containing pdb_files')
    parser.add_argument('--out_dir', type=str, required=True,
                        help='Output directory')
    parser.add_argument('--outname', type=str, default='CASF2016_scoring',
                        help='Output file name')
    
    parser.add_argument('--fragment', default=None, choices=['PS_300', 'PS_500'], help='Use fragment-based representation of small molecules')
    parser.add_argument('--interface_dist_th', type=float, default=10.0,
                        help='Residues who has atoms with distance below this threshold are considered in the complex interface')
    return parser.parse_args()


def process_line(line, struct_dir, interface_dist_th, fragment):
    line = line.strip().split(',')  # e.g. 4llx,2.89, 1
    assert len(line) == 3
    
    pdb_id, affinity_score, target_idx = line[0], float(line[1]), int(line[2])
    assert len(pdb_id) == 4
    
    # add target_idx (for ranking test)
    item = {
        'id': pdb_id,
        'target': target_idx
    }
    item['affinity'] = { 'neglog_aff': affinity_score }
    
    prot_fname = os.path.join(struct_dir, pdb_id, f'{pdb_id}_protein_pocket_{interface_dist_th}.pdb')
    sm_fname = os.path.join(struct_dir, pdb_id, f'{pdb_id}_ligand.sdf')

    try:
        list_blocks1 = pdb_to_list_blocks(prot_fname)
    except Exception as e:
        print_log(f'{pdb_id} protein parsing failed: {e}', level='ERROR')
        return None
    try:
        blocks2 = sdf_to_blocks(sm_fname, fragment=fragment)
    except Exception as e:
        print_log(f'{pdb_id} parsing failed: {e}', level='ERROR')
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


def process_single_line(line, struct_dir, interface_dist_th, use_fragment):
    """ process single line, return processed data  """
    item = process_line(line, struct_dir, interface_dist_th, use_fragment)
    if item in ('', None):  # drop invalid
        return None
    return item


def main(args):
    
    processed_pdbbind = []
    cnt = 0
    
    # load index file and preprocess
    print_log(f'Preprocessing ...')
    index_file = args.index_file
    struct_dir = args.benchmark_dir
    
    with open(index_file, 'r') as f:
        lines = f.readlines()[1:]  # skip headline
        
    # multi-process
    n_jobs = 48
    processed_pdbbind = Parallel(n_jobs=n_jobs)(
        delayed(process_single_line)(line, struct_dir, args.interface_dist_th, args.fragment is not None)
        for line in lines
    )

    # filter
    processed_pdbbind = [item for item in processed_pdbbind if item is not None]
    
    # summary
    print_log(f'valid/total = {len(processed_pdbbind)}/{len(lines)}')
    
    # save to file
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
        
    data_out_path = os.path.join(args.out_dir, f'{args.outname}.pkl')
    print_log(f'Obtained {len(processed_pdbbind)}, saving to {data_out_path}...')
    with open(data_out_path, 'wb') as fout:
        pickle.dump(processed_pdbbind, fout)

    print_log('Finished!')


if __name__ == '__main__':
    main(parse())
