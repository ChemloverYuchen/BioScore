#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import sys
import json
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
    parser = argparse.ArgumentParser(description='evaluate screening power')
    parser.add_argument('--test_dir', type=str, required=True, help='Path to the test dir')
    parser.add_argument('--dataset_info_file', type=str, required=True, help='Path to the dataset info file')
    parser.add_argument('--dataset_name', type=str, default='CASF2016', choices=['CASF2016', 'DEKOIS2.0_SP', 'DUD-E_SP','PDBbind_PP', 'pMHC'], help='Test dataset')
    parser.add_argument('--test_suffix', type=str, default=None, help='Test details')
    parser.add_argument('--family_name', type=str, default='all', help='Test protein family name')
    
    parser.add_argument('--task', type=str, default=None, choices=['PPI', 'PLI', 'PNI', 'NLI', 'PLI+PPI', 'PLI+PPI+PNI'])
    parser.add_argument('--output_type', type=str, default=None, choices=['docking/screening', 'scoring/ranking'], help='output type')
    
    # parser.add_argument('--epoch_idx', type=int, default=4, help='the ckpt for test')
    parser.add_argument('--topk_start', type=int, default=0, help='top-k ckpts for test: start idx')
    parser.add_argument('--topk_end', type=int, default=5, help='top-k ckpts for test: end idx')
    parser.add_argument('--start', type=int, required=True, help='start pdb_id idx')
    parser.add_argument('--end', type=int, required=True, help='end pdb_id idx')

    parser.add_argument('--results_dir', type=str, required=True, help='Path to the ckpt/results directory')

    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers to use')

    parser.add_argument('--gpu', type=int, default=-1, help='GPU to use, -1 for cpu')
    
    return parser.parse_args()


def main(args):
    
    # get pdb_ids
    with open(args.dataset_info_file, 'r') as f:
        pdb_id_ls = json.load(f)[args.dataset_name][args.family_name]
    print(len(pdb_id_ls))
   
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

            # load_data
            pdb_id_ls = pdb_id_ls[args.start:args.end]
            print(f'n_pdb: {len(pdb_id_ls)}')
            x = 0
            for pdb in pdb_id_ls:
                x += 1
                print(f'------------------------------ Processing pdb: {pdb}   {x}/{len(pdb_id_ls)}-----------------------------')
                # test_set_name = os.path.join(args.test_dir, f'{args.test_name}_{pdb}.pkl')
                if args.task == 'PPI':
                    test_set_name = os.path.join(args.test_dir, f'{args.dataset_name}_screening_6A', f'{args.dataset_name}_screening_6A_{pdb}.pkl')
                else:
                    test_set_name = os.path.join(args.test_dir, f'{args.dataset_name}_screening_10A', f'{args.dataset_name}_screening_10A_{pdb}.pkl')

                test_set = create_dataset(args.task, test_set_name)
                test_loader = DataLoader(test_set, batch_size=args.batch_size,
                                         num_workers=args.num_workers,
                                         collate_fn=test_set.collate_fn)
                items = test_set.indexes

                save_file_name = os.path.splitext(os.path.basename(ckpt_path))[0] + '_results.jsonl'
                if args.test_suffix is not None:
                    test_name = f'{args.dataset_name}_{args.test_suffix}_{args.family_name}'
                else:
                    test_name = f'{args.dataset_name}_{args.family_name}'

                save_dir = os.path.join(PROJ_DIR, args.results_dir, f'screening_results_{test_name}', pdb)

                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                save_path = os.path.join(save_dir, save_file_name)


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
                            binder = items[idx]['native']


                            # PDBbind screening benchmark
                            out_dict = {
                                    'id': item_id,
                                    'label': pred_label,   # energy  score....
                                    'task': args.task,
                                    'binder': binder
                                }

                            fout.write(json.dumps(out_dict) + '\n')
                            idx += 1

                fout.close()


if __name__ == '__main__':
    main(parse())
    
    