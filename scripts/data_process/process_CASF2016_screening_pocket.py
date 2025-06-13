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
                        help='Directory of all pdb dirs, each pdb dir: 57 * (5 + 280) * 100 decoys + info')
    parser.add_argument('--start', type=int, required=True,
                        help='start pdb_id idx')
    parser.add_argument('--end', type=int, required=True,
                        help='end pdb_id idx')
    
    parser.add_argument('--out_dir', type=str, required=True,
                        help='Output directory')
    parser.add_argument('--outname', type=str, default='CASF2016_screening',
                        help='Output file name')
    
    parser.add_argument('--fragment', default=None, choices=['PS_300', 'PS_500'], help='Use fragment-based representation of small molecules')
    parser.add_argument('--interface_dist_th', type=float, default=10.0,
                        help='Residues who has atoms with distance below this threshold are considered in the complex interface')
    return parser.parse_args()


def process_line(pdb_id, line, struct_dir, interface_dist_th, fragment):
    line = line.strip().split(',')  # e.g. 1e66-1a30_ligand_101,1e66,No
    assert len(line) == 3
    
    struct_idx, is_native = line[0], line[2]
    assert struct_idx.split('-')[0] == pdb_id
    
    print_log(f'Processing: {struct_idx}...')
    
    item = {
        'id': struct_idx,
        'native': is_native
    }
    item['affinity'] = { 'neglog_aff': 0.0 }
    
    cplx_id = struct_idx.split('_')[0].replace('-', '_')
    prot_fname = os.path.join(struct_dir, f'{pdb_id}_protein_pocket_{interface_dist_th}.pdb')
    sm_fname = os.path.join(struct_dir, cplx_id, f'{struct_idx}.sdf')

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
    """ process single line, return processed data """
    item = process_line(pdb_id, line, struct_dir, interface_dist_th, use_fragment)
    if item in ('', None):  # drop invalid
        return None
    return item


def process_pdb_id(pdb_id, root, args):
    """ process single pdb_id"""
    struct_dir = os.path.join(root, pdb_id)
    index_file = os.path.join(struct_dir, f'{pdb_id}_screening_info.csv')

    if not os.path.exists(index_file):
        print_log(f"Warning: {index_file} does not exist, skipping {pdb_id}")
        return 0

    # load index file
    with open(index_file, 'r') as f:
        lines = f.readlines()[1:]  # skip headline

    # multi-process
    n_jobs = 60
    processed_pdbbind = Parallel(n_jobs=n_jobs)(
        delayed(process_single_line)(pdb_id, line, struct_dir, args.interface_dist_th, args.fragment is not None)
        for line in lines
    )

    # filter
    processed_pdbbind = [item for item in processed_pdbbind if item is not None]

    # save to file
    data_out_path = os.path.join(args.out_dir, f'{args.outname}_{pdb_id}.pkl')
    print_log(f'Obtained {len(processed_pdbbind)}, saving to {data_out_path}...')
    with open(data_out_path, 'wb') as fout:
        pickle.dump(processed_pdbbind, fout)

    return len(processed_pdbbind)


def main(args):
    # get pdb_ids
    root = args.benchmark_dir
    # pdb_id_ls = sorted([pdb_id for pdb_id in os.listdir(root) if len(pdb_id) == 4])
    
    # screening pdb_ids（57）
    pdb_id_ls = ['1e66', '1mq6', '1nvq', '1o3f', '1sqa', '1u1b', '2al5', '2cet', '2fvd', '2p15',
             '2p4y', '2qbp', '2r9w', '2vvn', '2vw5', '2x00', '2xb8', '2yki', '2zcq', '3ag9',
             '3arp', '3coy', '3dd0', '3e93', '3ebp', '3ejr', '3f3e', '3fv1', '3g0w', '3ge7',
             '3gnw', '3kr8', '3myg', '3nw9', '3o9i', '3p5o', '3qqs', '3u8n', '3uex', '3uri',
             '3zso', '4agq', '4de1', '4f3c', '4gid', '4gr0', '4ivc', '4jia', '4pcs', '4rfm',
             '4tmn', '4twp', '4ty7', '4w9h', '5c2h', '5dwr', '3utu']
    print_log(len(pdb_id_ls))
    
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    
    total = 0
    for pdb_id in pdb_id_ls[args.start:args.end]:
        print_log(f'-------------------------Processing: {pdb_id}--------------------------')
        num_processed_items = process_pdb_id(pdb_id, root, args)
        total += num_processed_items
        
    print_log(f'Finished! Total: {total}')


if __name__ == '__main__':
    main(parse())
