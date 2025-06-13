import os
import sys
import json
import argparse
import numpy as np
import pandas as pd

from rdkit.ML.Scoring.Scoring import CalcBEDROC
from sklearn.metrics import roc_auc_score, average_precision_score

PROJ_DIR = os.path.abspath(os.path.join(
    os.path.split(os.path.abspath(__file__))[0],
    '..', '..'
))
sys.path.append(PROJ_DIR)
print(PROJ_DIR)

def parse():
    parser = argparse.ArgumentParser(description='evaluate screening power')    
    parser.add_argument('--results_dir', type=str, required=True, help='Path to the results directory')
    parser.add_argument('--dataset_info_file', type=str, required=True, help='Path to the dataset info file')
    
    parser.add_argument('--dataset_name', type=str, default='CASF2016', choices=['CASF2016', 'DEKOIS2.0_SP', 'DUD-E_SP','PDBbind_PP','pMHC'], help='Test dataset')
    parser.add_argument('--test_suffix', type=str, default=None, help='Test details')
    parser.add_argument('--family_name', type=str, default='all', help='Test protein family name')
    
    # parser.add_argument('--epoch_idx', type=int, default=0, help='the ckpt for test')
    parser.add_argument('--topk_start', type=int, default=0, help='top-k ckpts for test: start idx')
    parser.add_argument('--topk_end', type=int, default=5, help='top-k ckpts for test: end idx')
    
    parser.add_argument('--alpha', type=float, default=80.5, help='Percentage threshold for evaluation (0 < alpha <= 100)')
    parser.add_argument('--k1', type=float, default=0.005, help='Percentage threshold for evaluation (0 < k <= 1)')
    parser.add_argument('--k2', type=float, default=0.01, help='Percentage threshold for evaluation (0 < k <= 1)')
    parser.add_argument('--k3', type=float, default=0.05, help='Percentage threshold for evaluation (0 < k <= 1)')
 
    return parser.parse_args()


def compute_bedroc(labels, scores, alpha=80.5):
    labels = np.array(labels)  # gt
    scores = np.array(scores)  # prediction
    
    sort_ind = np.argsort(-scores)  # sorted by prediction
    bedroc = CalcBEDROC(labels[sort_ind].reshape(-1, 1), 0, alpha=80.5)
    
    return bedroc

def compute_ef(labels, scores, top_ratio=0.005):
    labels = np.array(labels)
    scores = np.array(scores)
    n = len(labels)
    n_top = max(1, int(top_ratio * n))

    sorted_indices = np.argsort(-scores)
    top_indices = sorted_indices[:n_top]

    ef = np.sum(labels[top_indices]) / (np.sum(labels) * top_ratio + 1e-8)
    return ef

def compute_aupr(labels, scores):
    try:
        return average_precision_score(labels, scores)
    except:
        return np.nan

def obtain_metircs(df, alpha=80.5, k1=0.005, k2=0.01, k3=0.05):
    # calculate metrics by target
    metric_dict = {
        'AUROC': [],
        'AUPR': [],
        'BEDROC': [],
        f'EF top-{float(k1 * 100)}': [],
        f'EF top-{float(k2 * 100)}': [],
        f'EF top-{float(k3 * 100)}': [],
    }
    
    for target, group in df.groupby('target'):
        labels = (group['binder'] == 'Yes').astype(int).values
        scores = group['label'].values

        if labels.sum() == 0 or labels.sum() == len(labels):
            continue
        
        try:
            auroc = roc_auc_score(labels, scores)
        except:
            auroc = np.nan
        aupr = compute_aupr(labels, scores)
        bedroc = compute_bedroc(labels, scores, alpha=alpha)
        ef_k1 = compute_ef(labels, scores, top_ratio=k1)
        ef_k2 = compute_ef(labels, scores, top_ratio=k2)
        ef_k3 = compute_ef(labels, scores, top_ratio=k3)
     
        metric_dict['AUROC'].append(auroc)
        metric_dict['AUPR'].append(aupr)
        metric_dict['BEDROC'].append(bedroc)
        metric_dict[f'EF top-{float(k1 * 100)}'].append(ef_k1)
        metric_dict[f'EF top-{float(k2 * 100)}'].append(ef_k2)
        metric_dict[f'EF top-{float(k3 * 100)}'].append(ef_k3)
    
    return metric_dict

def main(args):
    alpha = args.alpha
    k1, k2, k3 = args.k1, args.k2, args.k3
    metrics = ['AUROC_mean', 'AUROC_median', 'AUPR_mean', 'AUPR_median', 'BEDROC_mean', 'BEDROC_median',
           f'EF top-{float(k1 * 100):.1f}_mean', f'EF top-{float(k1 * 100):.1f}_median',
           f'EF top-{float(k2 * 100):.1f}_mean', f'EF top-{float(k2 * 100):.1f}_median',
           f'EF top-{float(k3 * 100):.1f}_mean', f'EF top-{float(k3 * 100):.1f}_median']
    
    d = {'ckpt': []}
    for metric in metrics:
        d[metric] = []
        
    # extract pdb_ids
    with open(args.dataset_info_file, 'r') as f:
        pdb_id_ls = json.load(f)[args.dataset_name][args.family_name]
    print(f'{args.dataset_name} {args.family_name}: {len(pdb_id_ls)}')
  
    version = args.results_dir.split('/')[-1]
    topk_map_path = os.path.join(PROJ_DIR, args.results_dir, 'checkpoint', 'topk_map.txt')
    
    with open(topk_map_path, 'r') as f:
        lines = f.readlines()
    
    for i, line in enumerate(lines[args.topk_start:args.topk_end]):
        epoch_idx = args.topk_start + i
        metric, ckpt_path = line.strip('\n').split(': ')
        ckpt_path = os.path.join(PROJ_DIR, ckpt_path)
        file_name = os.path.splitext(os.path.basename(ckpt_path))[0] + '_results.jsonl'

        if args.test_suffix is not None:
            test_name = f'{args.dataset_name}_{args.test_suffix}_{args.family_name}'
        else:
            test_name = f'{args.dataset_name}_{args.family_name}'

        score_df_ls = []
        for pdb in pdb_id_ls:
            print(f'------------------------ {pdb}--------------------------')

            # load json file
            file_path = os.path.join(PROJ_DIR, args.results_dir, f'screening_results_{test_name}', pdb, file_name)
            
            data = pd.read_json(file_path, lines=True)
            
            # select the highest-scoring entry per group
            data['group_id'] = data['id'].apply(lambda x: x.split('_')[0])
            score_df = data.loc[data.groupby('group_id')['label'].idxmax()]
            score_df['target'] = pdb
            score_df_ls.append(score_df)
        combined_df = pd.concat(score_df_ls, axis=0)

        metric_dict = obtain_metircs(combined_df, alpha=alpha, k1=k1, k2=k2, k3=k3)

        details_dict = {}
        details_dict['pdb_id'] = pdb_id_ls
        for key in metric_dict:
            details_dict[key] = [round(score, 4) for score in metric_dict[key]]
        details_df = pd.DataFrame(details_dict)
        details_file = os.path.join(PROJ_DIR, args.results_dir, f'{version}_screening_results_{test_name}_details_top{epoch_idx}.csv')
        details_df.to_csv(details_file, index=False)


        # calculate metrics
        for key in metric_dict:
            values = [v for v in metric_dict[key] if not np.isnan(v)]
            d[f'{key}_mean'].append(round(np.mean(values), 4))
            d[f'{key}_median'].append(round(np.median(values), 4))

        d['ckpt'].append(os.path.splitext(os.path.basename(ckpt_path))[0])

        # results
        print("\nEvaluation Results Across All PDBs:")
        print("-" * 50)

        print(f"AUROC Mean  : {d['AUROC_mean'][0]:.4f}")
        print(f"AUROC Median : {d['AUROC_median'][0]:.4f}")
        print(f"AUPR Mean  : {d['AUPR_mean'][0]:.4f}")
        print(f"AUPR Median : {d['AUPR_median'][0]:.4f}")
        print(f"BEDROC Mean  : {d['BEDROC_mean'][0]:.4f}")
        print(f"BEDROC Median: {d['BEDROC_median'][0]:.4f}")

        print(f"EF top-{float(k1 * 100):.1f} Mean   : {d[f'EF top-{float(k1 * 100):.1f}_mean'][0]:.4f}")
        print(f"EF top-{float(k1 * 100):.1f} Median : {d[f'EF top-{float(k1 * 100):.1f}_median'][0]:.4f}")
        print(f"EF top-{float(k2 * 100):.1f} Mean   : {d[f'EF top-{float(k2 * 100):.1f}_mean'][0]:.4f}")
        print(f"EF top-{float(k2 * 100):.1f} Median : {d[f'EF top-{float(k2 * 100):.1f}_median'][0]:.4f}")
        print(f"EF top-{float(k3 * 100):.1f} Mean   : {d[f'EF top-{float(k3 * 100):.1f}_mean'][0]:.4f}")
        print(f"EF top-{float(k3 * 100):.1f} Median : {d[f'EF top-{float(k3 * 100):.1f}_median'][0]:.4f}")

        print("-" * 50)

    results_df = pd.DataFrame(d)
    results_file = os.path.join(PROJ_DIR, args.results_dir, f'{version}_screening_results_{test_name}.csv')
    results_df.to_csv(results_file, index=False)

if __name__ == '__main__':
    main(parse())
