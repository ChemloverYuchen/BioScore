#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import sys
import argparse
import json
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

PROJ_DIR = os.path.abspath(os.path.join(
    os.path.split(os.path.abspath(__file__))[0],
    '..', '..'
))
sys.path.append(PROJ_DIR)
print(PROJ_DIR)

import models
from train import create_dataset
from data.pdb_utils import VOCAB


def parse():
    parser = argparse.ArgumentParser(description='evaluate docking power')
    parser.add_argument('--test_set', type=str, required=True, help='Path to the test set')
    parser.add_argument('--task', type=str, default=None, choices=['PPI', 'PLI', 'PNI', 'NLI', 'PLI+PPI', 'PLI+PPI+PNI'])
    
    parser.add_argument('--topk_start', type=int, default=0, help='top-k ckpts for test: start idx')
    parser.add_argument('--topk_end', type=int, default=5, help='top-k ckpts for test: end idx')
    
    parser.add_argument('--results_dir', type=str, required=True, help='Path to the ckpt/results directory')
    parser.add_argument('--test_name', type=str, default='PDBbind', help='Test dataset')
    parser.add_argument('--criterion', type=str, default='rmsd', help='criterion for test')

    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers to use')
    parser.add_argument('--output_type', type=str, default=None, choices=['docking/screening', 'scoring/ranking'], help='output type')
    parser.add_argument('--gpu', type=int, default=-1, help='GPU to use, -1 for cpu')

    
    return parser.parse_args()


def main(args):
    
    # load data
    test_set = create_dataset(args.task, args.test_set)
    test_loader = DataLoader(test_set, batch_size=args.batch_size,
                             num_workers=args.num_workers,
                             collate_fn=test_set.collate_fn)
    items = test_set.indexes
    
    # load model     
    ckpt_dir = os.path.join(PROJ_DIR, args.results_dir, 'checkpoint')
    namespace = json.load(open(os.path.join(ckpt_dir, 'namespace.json'), 'r'))
    train_args = argparse.Namespace(**namespace)
    device = torch.device('cpu' if args.gpu == -1 else f'cuda:{args.gpu}')
    
    topk_map_path = os.path.join(ckpt_dir, 'topk_map.txt')
    with open(topk_map_path, 'r') as f:
        lines = f.readlines()
        
    for line in lines[args.topk_start:args.topk_end]:
        metric, ckpt_path = line.strip('\n').split(': ')
        ckpt_path = os.path.join(PROJ_DIR, ckpt_path)
        save_file_name = os.path.splitext(os.path.basename(ckpt_path))[0] + '_results.jsonl'
        save_dir = os.path.join(PROJ_DIR, args.results_dir, f'docking_results_{args.test_name}')
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        save_path = os.path.join(save_dir, save_file_name)
        print(ckpt_path)
        
        model = torch.load(ckpt_path, map_location='cpu')
    
        if not isinstance(model, torch.nn.Module):
            weights = model
            model = models.create_model(train_args)
            model.load_state_dict(weights)
        else:
            print("do not need to modify code for test")

        model.to(device)
        model.eval()
        
        criterion_name = args.criterion

        fout = open(save_path, 'w')

        idx = 0
        for batch in tqdm(test_loader):
            with torch.no_grad():
                # move data
                for k in batch:
                    if hasattr(batch[k], 'to'):
                        batch[k] = batch[k].to(device)
                del batch['label']

                results = model.infer(batch, args.output_type)
                if type(results) == tuple:

                    results = (res.tolist() for res in results)
                    results = (res for res in zip(*results))
                else:
                    results = results.tolist()

                for pred_label in results:
                    item_id = items[idx]['id']
                    
                    # docking benchmark
                    criterion = items[idx][criterion_name]
                    out_dict = {
                            'id': item_id,
                            'label': pred_label,   # energy  score....
                            'task': args.task,
                            criterion_name: criterion
                        }

                    fout.write(json.dumps(out_dict) + '\n')
                    idx += 1

        fout.close()


if __name__ == '__main__':
    main(parse())
    
    