#!/usr/bin/python
# -*- coding:utf-8 -*-
import re
import os
import sys
import json
import argparse
from argparse import Namespace
from multiprocessing import Process

import numpy as np
import subprocess

PROJ_DIR = os.path.abspath(os.path.join(
    os.path.split(os.path.abspath(__file__))[0],
    '..', '..'
))
sys.path.append(PROJ_DIR)

from utils.logger import print_log


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                        help='Path of the configuration for training the model')
    parser.add_argument('--gpus', type=int, nargs='+', default=[0], help='gpu to use, -1 for cpu')
    return parser.parse_args()


def main(args):

    # 1. Load configuration and save to temporary file
    config: dict = json.load(open(args.config, 'r'))
    print_log(f'General configuration: {config}')

    # 2. get configures and delete training-unrelated configs
    out_dir = config.pop('out_dir')
    tmp_config_path = os.path.join(out_dir, 'tmp_config.json')

    # 3. create output directory
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # 4. experiments
    seeds = [42]
    round_metrics = []
    for i, seed in enumerate(seeds):
        print()
        print_log(f'Start the {i}-th experiment with seed {seed}')

        # 5. add data path / save directory to config
        config['seed'] = seed

        # training
        print_log(f'Configuration: {config}')
        print_log('Start training')
        json.dump(config, open(tmp_config_path, 'w'))
        
        p = subprocess.Popen(
            f'GPU={",".join([str(gpu) for gpu in args.gpus])} bash {PROJ_DIR}/scripts/train/train.sh {tmp_config_path}',
            shell=True,
            stdout=sys.stdout,
            stderr=sys.stderr
        )
        p.wait()
        
if __name__ == '__main__':
    import multiprocessing
    multiprocessing.set_start_method('spawn')
    print(f'Project directory: {PROJ_DIR}')
    main(parse())