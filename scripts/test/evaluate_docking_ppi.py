import os
import sys
import pandas as pd
from scipy.stats import spearmanr, kendalltau
import argparse

PROJ_DIR = os.path.abspath(os.path.join(
    os.path.split(os.path.abspath(__file__))[0],
    '..', '..'
))
sys.path.append(PROJ_DIR)


def parse():
    parser = argparse.ArgumentParser(description='evaluate docking power')
    parser.add_argument('--results_dir', type=str, required=True, help='Path to the results directory')
    parser.add_argument('--test_name', type=str, default='PDBbind', help='Test dataset')
    
    parser.add_argument('--topk_start', type=int, default=0, help='top-k ckpts for test: start idx')
    parser.add_argument('--topk_end', type=int, default=5, help='top-k ckpts for test: end idx')
    
    parser.add_argument('--dockq', type=float, default=0.8, help='dockq threshold for test')

    parser.add_argument('--k1', type=int, default=1, help='top-k for success evaluation')
    parser.add_argument('--k2', type=int, default=5, help='top-k for success evaluation')
    parser.add_argument('--k3', type=int, default=10, help='top-k for success evaluation')
 
    return parser.parse_args()

# Calculate Spearman & Kendall's Tau
def compute_rank_correlations(group, k, rmsd_cut=0.8):
    # ranking(label, dockq)
    label_rank = group['label'].rank(ascending=False)
    rmsd_rank = group['dockq'].rank(ascending=False)

    # All Spearman/Kendall's Tau
    spearman_corr, _ = spearmanr(label_rank, rmsd_rank)
    kendall_corr, _ = kendalltau(label_rank, rmsd_rank)

    # dockq top-20% Spearman/Kendall's Tau
    n_top20 = 20
    group_top20 = group.sort_values(by='dockq', ascending=False).head(n_top20)
    top20_label_rank = group_top20['label'].rank(ascending=False)
    top20_rmsd_rank = group_top20['dockq'].rank(ascending=False)
    top20_spearman_corr, _ = spearmanr(top20_label_rank, top20_rmsd_rank)
    top20_kendall_corr, _ = kendalltau(top20_label_rank, top20_rmsd_rank)

    # top-1 (dockq = 1)
    rmsd_rank_1_samples = group[rmsd_rank == 1]

    # success: according to label_rank index of top-k
    top_k_indices = label_rank.nsmallest(k).index
    label_rank_threshold = label_rank.loc[top_k_indices].max()
    success = any(rmsd_rank_1_samples.index.isin(top_k_indices))

    # rank_percentiles
    rank_percentiles = group['label'].rank(pct=True, ascending=True)[rmsd_rank == 1].values

    # hit rate: dockq >= 0.8 in top-k
    top_indices = group.nlargest(k, 'label').index
    success_rmsd = group.dockq.loc[top_indices].max() >= rmsd_cut

    return pd.Series({
        'spearman_corr': spearman_corr,
        'kendall_corr': kendall_corr,
        'top20_spearman_corr': top20_spearman_corr,
        'top20_kendall_corr': top20_kendall_corr,
        'success': success,
        'success_rmsd': success_rmsd,
        'label': rmsd_rank_1_samples['label'].values,
        'rmsd': rmsd_rank_1_samples['dockq'].values,
        'label_rank': label_rank[rmsd_rank == 1].values,
        'label_rank_threshold': label_rank_threshold,
        'percentile': rank_percentiles
    })

def main(args):
    version = args.results_dir.split('/')[-1]
    topk_map_path = os.path.join(PROJ_DIR, args.results_dir, 'checkpoint', 'topk_map.txt')
    
    with open(topk_map_path, 'r') as f:
        lines = f.readlines()
    
    k1, k2, k3 = args.k1, args.k2, args.k3
    d = {'ckpt': [], 'Mean Spearman': [], 'Mean Kendall': [], 'Top20 Spearman': [], 'Top20 Kendall': [],
         f'Success Rate top-{k1}': [], f'Success Rate top-{k2}': [], f'Success Rate top-{k3}': [],
         f'Hit Rate top-{k1}': [], f'Hit Rate top-{k2}': [], f'Hit Rate top-{k3}': []}
        
    for line in lines[args.topk_start:args.topk_end]:
        metric, ckpt_path = line.strip('\n').split(': ')
        file_name = os.path.splitext(os.path.basename(ckpt_path))[0] + '_results.jsonl'
        file_path = os.path.join(PROJ_DIR, args.results_dir, f'docking_results_{args.test_name}', file_name)
        
        print(f'{file_name} docking results:')
        
        # load json file
        data = pd.read_json(file_path, lines=True)
        
        # extract pdb_id, e.g. 1d6r_2_6
        data['pdb'] = data['id'].apply(lambda x: x.split('_')[0])

        # filtered_data = data.groupby('pdb').filter(lambda x: len(x) >= 3)
        filtered_data = data.groupby('pdb').filter(lambda x: len(x) >= 10)

        # group by pdb_id
        rmsd_cut = args.dockq
        correlations_1 = filtered_data.groupby('pdb').apply(compute_rank_correlations, k=k1, rmsd_cut=rmsd_cut)
        correlations_2 = filtered_data.groupby('pdb').apply(compute_rank_correlations, k=k2, rmsd_cut=rmsd_cut)
        correlations_3 = filtered_data.groupby('pdb').apply(compute_rank_correlations, k=k3, rmsd_cut=rmsd_cut)

        # calculate success_rate
        success_rate_1 = correlations_1['success'].mean()
        success_rate_2 = correlations_2['success'].mean()
        success_rate_3 = correlations_3['success'].mean()
        
        # calculate hit_rate（dockq>=0.8）
        success_rate_rmsd_1 = correlations_1['success_rmsd'].mean()
        success_rate_rmsd_2 = correlations_2['success_rmsd'].mean()
        success_rate_rmsd_3 = correlations_3['success_rmsd'].mean()

        # average
        mean_spearman = correlations_1['spearman_corr'].mean()
        mean_kendall = correlations_1['kendall_corr'].mean()
        top20_spearman = correlations_1['top20_spearman_corr'].mean()
        top20_kendall = correlations_1['top20_kendall_corr'].mean()

        # results
        print("Per PDB Correlations and Success Rate:")
        print(correlations_1.to_string())

        print("\nMean Correlations Across All PDBs:")
        print(f"Mean Spearman Correlation: {mean_spearman:.4f}")
        print(f"Mean Kendall Correlation: {mean_kendall:.4f}")
        print(f"Top20 Spearman Correlation: {top20_spearman:.4f}")
        print(f"Top20 Kendall Correlation: {top20_kendall:.4f}")
        print(f"Success Rate top-{k1}: {success_rate_1 * 100:.2f}%")
        print(f"Success Rate top-{k2}: {success_rate_2 * 100:.2f}%")
        print(f"Success Rate top-{k3}: {success_rate_3 * 100:.2f}%")
        print(f"Hit Rate top-{k1}: {success_rate_rmsd_1 * 100:.2f}%")
        print(f"Hit Rate top-{k2}: {success_rate_rmsd_2 * 100:.2f}%")
        print(f"Hit Rate top-{k3}: {success_rate_rmsd_3 * 100:.2f}%")
        print('-' * 30)
        
        d['ckpt'].append(os.path.splitext(os.path.basename(ckpt_path))[0])
        d['Mean Spearman'].append(round(mean_spearman, 4))
        d['Mean Kendall'].append(round(mean_kendall, 4))
        d['Top20 Spearman'].append(round(top20_spearman, 4))
        d['Top20 Kendall'].append(round(top20_kendall, 4))
        d[f'Success Rate top-{k1}'].append(round(success_rate_1 * 100, 2))
        d[f'Success Rate top-{k2}'].append(round(success_rate_2 * 100, 2))
        d[f'Success Rate top-{k3}'].append(round(success_rate_3 * 100, 2))
        d[f'Hit Rate top-{k1}'].append(round(success_rate_rmsd_1 * 100, 2))
        d[f'Hit Rate top-{k2}'].append(round(success_rate_rmsd_2 * 100, 2))
        d[f'Hit Rate top-{k3}'].append(round(success_rate_rmsd_3 * 100, 2))

    results_df = pd.DataFrame(d)
    results_file = os.path.join(PROJ_DIR, args.results_dir, f'{version}_docking_results_{args.test_name}.csv')
    results_df.to_csv(results_file, index=False)

    
if __name__ == '__main__':
    main(parse())
