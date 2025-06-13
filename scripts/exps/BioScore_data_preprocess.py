#!/usr/bin/python
# -*- coding:utf-8 -*-
import re
import gc
import os
import sys
import json
import argparse
from argparse import Namespace
from multiprocessing import Process
import subprocess

import numpy as np
# import torch

PROJ_DIR = os.path.abspath(os.path.join(
    os.path.split(os.path.abspath(__file__))[0],
    '..', '..'
))
# print(f'Project directory: {PROJ_DIR}')
sys.path.append(PROJ_DIR)

from data.split import main as split
from data.dataset import BlockGeoAffDataset, NLIDataset

from utils.logger import print_log


def parse():
    parser = argparse.ArgumentParser(description='Process PDBbind')
    parser.add_argument('--split_dir', type=str, required=True,
                        help='Path to data containing train/valid.pkl')
    parser.add_argument('--data_type', type=str, required=True,
                        help='data_type: PPI/PNI/NLI')
    return parser.parse_args()


def main(args):
    print_log(f'Preprocess {args.split_dir}...')
    for split_name in ['valid', 'train']:
        if args.data_type == 'PPI' or args.data_type == 'PNI':
            dataset = BlockGeoAffDataset(os.path.join(args.split_dir, f'{split_name}.pkl'))
        elif args.data_type == 'NLI':
            dataset = NLIDataset(os.path.join(args.split_dir, f'{split_name}.pkl'))
        else:
            dataset = []
        print_log(f'{split_name} lengths: {len(dataset)}')
        

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.set_start_method('spawn')
    print(f'Project directory: {PROJ_DIR}')
    main(parse())
