import os
import sys
import json
import pandas as pd
from scipy.stats import spearmanr, kendalltau
import argparse


PROJ_DIR = os.path.abspath(os.path.join(
    os.path.split(os.path.abspath(__file__))[0],
    '..', '..'
))
sys.path.append(PROJ_DIR)
print(PROJ_DIR)


def parse():
    parser = argparse.ArgumentParser(description='evaluate docking power')
    parser.add_argument('--results_dir', type=str, required=True, help='Path to the results directory')
    parser.add_argument('--dataset_info_file', type=str, required=True, help='Path to the dataset info file')
    
    parser.add_argument('--dataset_name', type=str, default='CASF2016', choices=['PDBbind_PP'], help='Test dataset')
    parser.add_argument('--test_suffix', type=str, default=None, help='Test details')
    parser.add_argument('--family_name', type=str, default='all', help='Test protein family name')
    
    # parser.add_argument('--epoch_idx', type=int, default=4, help='the ckpt for test')
    parser.add_argument('--topk_start', type=int, default=0, help='top-k ckpts for test: start idx')
    parser.add_argument('--topk_end', type=int, default=5, help='top-k ckpts for test: end idx')
    
    parser.add_argument('--k1', type=float, default=0.01, help='Percentage threshold for success evaluation (0 < k <= 1)')
    parser.add_argument('--k2', type=float, default=0.05, help='Percentage threshold for success evaluation (0 < k <= 1)')
    parser.add_argument('--k3', type=float, default=0.10, help='Percentage threshold for success evaluation (0 < k <= 1)')
 
    return parser.parse_args()


def compute_success_rate(group, k):
    # soreted by label
    label_rank = group['label'].rank(ascending=False)
     
    # get true binder
    true_binder_samples = group[group['binder'] == 'Yes']
    
    label_rank_threshold = label_rank.quantile(k)
    
    success = any(label_rank[true_binder_samples.index] <= label_rank_threshold)
    
    return pd.Series({
        'success': success,
        'label': true_binder_samples['label'].values ,
        'label_rank': label_rank[true_binder_samples.index].values,
    })


def main(args):
#     clusters = {
#     3: ['2omz', '2omw'],
#     7: ['3ul1', '3ukz', '6iu7', '6iua', '1pjn', '3ul0', '1pjm', '3ukx', '6k06'],
#     18: ['5ky4', '5ky5'],
#     21: ['3vv2', '2z58'],
#     35: ['1lw6', '2sic'],
#     43: ['2gww', '1syq', '1t01', '1rkc'],
#     67: ['1kac', '1p69'],
#     78: ['1gua', '3kuc'],
#     110: ['1emv', '2vln', '2vlp']
#     }
    
#     id_to_clusters = {}
#     for cluster_id, ids in clusters.items():
#         for id in ids:
#             if id not in id_to_clusters:
#                 id_to_clusters[id] = cluster_id 
    
    
    # metrics
    k1, k2, k3 = args.k1, args.k2, args.k3
    d = {'ckpt': [],
       f'Success Rate top-{int(k1 * 100)}(%)': [],
       f'Success Rate top-{int(k2 * 100)}(%)': [],
       f'Success Rate top-{int(k3 * 100)}(%)': [],
       'average label rank': []}
    
      
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

        combined_df = pd.DataFrame()
        for pdb in pdb_id_ls:
            file_path = os.path.join(PROJ_DIR, args.results_dir, f'screening_results_{test_name}', pdb, file_name)

            # load json file
            data = pd.read_json(file_path, lines=True)
            data['binder'] = data['binder'].apply(lambda x: x.strip('\n'))

            # select the highest-scoring entry per group
            data['group_id'] = data['id'].apply(lambda x: x.split('_')[0])
            data['rec'] = data['group_id'].apply(lambda x: x.split('-')[0])
            data['lig'] = data['group_id'].apply(lambda x: x.split('-')[1])
            max_score_df = data.loc[data.groupby('group_id')['label'].idxmax()]
            
#             # drop false negative binders
#             max_score_df['to_remove'] = max_score_df.apply(
#                 lambda row: (
#                     (row['rec'] != row['lig']) and
#                     (row['rec'] in id_to_clusters) and
#                     (row['lig'] in id_to_clusters) and
#                     (id_to_clusters[row['rec']] == id_to_clusters[row['lig']])
#                 ), axis=1)

#             max_score_df_filtered = max_score_df[~max_score_df['to_remove']]

#             max_score_df = max_score_df_filtered.drop(columns=['to_remove'])
            
            combined_df = pd.concat([combined_df, max_score_df], ignore_index=True)

        final_result_dir = os.path.join(PROJ_DIR, args.results_dir, f'screening_results_{test_name}', 'all')
        if not os.path.exists(final_result_dir):
            os.makedirs(final_result_dir)
        final_file = os.path.join(final_result_dir, file_name)

        combined_df.to_json(final_file, orient='records', lines=True)
        print(len(combined_df))

        # calculate success rate by group
        print(f'{file_name} screening results:')
        combined_df['pdb_id'] = combined_df['group_id'].apply(lambda x: x.split('-')[0])

        # group by pdb_id
        correlations_1 = combined_df.groupby('pdb_id').apply(compute_success_rate, k=k1)
        correlations_2 = combined_df.groupby('pdb_id').apply(compute_success_rate, k=k2)
        correlations_3 = combined_df.groupby('pdb_id').apply(compute_success_rate, k=k3)

        # calculate success rate
        success_rate_1 = correlations_1['success'].mean()
        success_rate_2 = correlations_2['success'].mean()
        success_rate_3 = correlations_3['success'].mean()

        # results
        per_pdb_summary_file = os.path.join(PROJ_DIR, args.results_dir, f'{version}_screening_results_{test_name}_per_pdb_top{epoch_idx}.csv')
        summary_df = pd.DataFrame()
        summary_df['pdb_id'] = correlations_1.index
        summary_df['label'] = [item[0] for item in correlations_1['label'].tolist()]
        summary_df['label_rank'] = [int(item[0]) for item in correlations_1['label_rank'].tolist()]

        summary_df.to_csv(per_pdb_summary_file, index=False)

        print("Per PDB Correlations and Success Rate:")
        print(correlations_1.to_string())

        print("\nSuccess Rate Across All PDBs:")
        print(f"Success Rate top-{int(k1 * 100)}(%): {success_rate_1 * 100:.2f}%")
        print(f"Success Rate top-{int(k2 * 100)}(%): {success_rate_2 * 100:.2f}%")
        print(f"Success Rate top-{int(k3 * 100)}(%): {success_rate_3 * 100:.2f}%")
        print(f"average label rank: {summary_df['label_rank'].mean()}")
        print('-' * 30)

        d['ckpt'].append(os.path.splitext(os.path.basename(ckpt_path))[0])
        d[f'Success Rate top-{int(k1 * 100)}(%)'].append(round(success_rate_1 * 100, 2))
        d[f'Success Rate top-{int(k2 * 100)}(%)'].append(round(success_rate_2 * 100, 2))
        d[f'Success Rate top-{int(k3 * 100)}(%)'].append(round(success_rate_3 * 100, 2))
        d['average label rank'].append(round(summary_df['label_rank'].mean(), 2))

    results_df = pd.DataFrame(d)
    results_file = os.path.join(PROJ_DIR, args.results_dir, f'{version}_screening_results_{test_name}.csv')
    results_df.to_csv(results_file, index=False)

    
if __name__ == '__main__':
    main(parse())
