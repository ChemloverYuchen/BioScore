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
print(PROJ_DIR)


def parse():
    parser = argparse.ArgumentParser(description='evaluate docking power')
    parser.add_argument('--casf_root', type=str, default='/data1/baidu/data/protein-mol/CASF-2016/power_screening', help='Path to the CASF directory')
    
    parser.add_argument('--results_dir', type=str, required=True, help='Path to the results directory')
    parser.add_argument('--test_name', type=str, default='CASF2016', help='Test dataset')
    
    # parser.add_argument('--epoch_idx', type=int, default=0, help='the ckpt for test')
    parser.add_argument('--topk_start', type=int, default=0, help='top-k ckpts for test: start idx')
    parser.add_argument('--topk_end', type=int, default=5, help='top-k ckpts for test: end idx')
    
    parser.add_argument('--k1', type=float, default=0.01, help='Percentage threshold for evaluation (0 < k <= 1)')
    parser.add_argument('--k2', type=float, default=0.05, help='Percentage threshold for evaluation (0 < k <= 1)')
    parser.add_argument('--k3', type=float, default=0.10, help='Percentage threshold for evaluation (0 < k <= 1)')
 
    return parser.parse_args()


# success_rate: L1 in top-k%
def calc_success_rate(df_groupby, score_name, top=0.01):
    total_mol = len(df_groupby)
    success_mol = df_groupby.apply(lambda x: 1 if x.topactid.values[0] in x.nlargest(round(top * len(x)),score_name).index else 0,
                                   include_groups=False).sum()
    return success_mol, success_mol/total_mol


def calc_ef(df_total, score_name, label_name, threshold=0.01):
    N_total = len(df_total)
    N_actives = len(df_total[df_total[label_name] == 1])
    total_sorted = df_total.sort_values(by=[score_name], ascending=False)
    
    N_topx_total = round(N_total * threshold)
    topx_total = total_sorted.iloc[:N_topx_total, :]
    N_topx_actives = len(topx_total[topx_total[label_name] == 1])
    
    return N_topx_actives / (N_actives * threshold)


def obtain_metircs(df, k1=0.01, k2=0.05, k3=0.10):
    df_groupby = df.groupby('target')
    topnum1, SR1 = calc_success_rate(df_groupby, 'score', top=k1)
    topnum5, SR5 = calc_success_rate(df_groupby, 'score', top=k2)
    topnum10, SR10 = calc_success_rate(df_groupby, 'score', top=k3)

    EF1 = df_groupby.apply(lambda x: calc_ef(x, 'score', 'label', threshold=k1), include_groups=False).mean()
    EF5 = df_groupby.apply(lambda x: calc_ef(x, 'score', 'label', threshold=k2), include_groups=False).mean()
    EF10 = df_groupby.apply(lambda x: calc_ef(x, 'score', 'label', threshold=k3), include_groups=False).mean()
    
    return topnum1, SR1*100, topnum5, SR5*100, topnum10, SR10*100, EF1, EF5, EF10


def main(args):
    # load target-ligand info
    core_file = os.path.join(args.casf_root, 'CoreSet.dat')
    core_df = pd.read_csv(core_file, sep='[,,_,\t, ]+', header=0, engine='python')
    topligs = core_df.groupby('target').apply(lambda x: x.sort_values('logKa').iloc[-1], include_groups=False)
    
    target_file = os.path.join(args.casf_root, 'TargetInfo.dat')
    target_df = pd.read_csv(target_file, sep='[,,\t, ]+', header=0, skiprows=8, engine='python')
    target_df = target_df.drop_duplicates(subset=['#T'], keep='first')
    target_df['target'] = [int(core_df[core_df['#code'] == i].target.iloc[0]) for i in target_df['#T']]
    print(len(target_df))
    
    
    version = args.results_dir.split('/')[-1]
    topk_map_path = os.path.join(PROJ_DIR, args.results_dir, 'checkpoint', 'topk_map.txt')
    
    with open(topk_map_path, 'r') as f:
        lines = f.readlines()
        
    k1, k2, k3 = args.k1, args.k2, args.k3
    d = {'ckpt': [],
       f'SR top-{int(k1 * 100)}(%)': [], f'SR top-{int(k2 * 100)}(%)': [], f'SR top-{int(k3 * 100)}(%)': [],
       f'EF top-{int(k1 * 100)}': [],   f'EF top-{int(k2 * 100)}': [],   f'EF top-{int(k3 * 100)}': []}

    results_root = os.path.join(PROJ_DIR, args.results_dir, f'screening_results_{args.test_name}')
    pdb_id_ls = sorted([pdb_id for pdb_id in os.listdir(results_root) if len(pdb_id) == 4])
    
    for i, line in enumerate(lines[args.topk_start:args.topk_end]):
        metric, ckpt_path = line.strip('\n').split(': ')
        ckpt_path = os.path.join(PROJ_DIR, ckpt_path)
        file_name = os.path.splitext(os.path.basename(ckpt_path))[0] + '_results.jsonl'

        score_df_ls = []
        for pdb in pdb_id_ls:
            print(pdb)
            act_id = target_df[target_df['#T'] == pdb].iloc[:, 1:-1].dropna(axis=1).values.reshape(-1)
            topact_id = target_df[target_df['#T'] == pdb].loc[:, 'L1'].values[0]

            # load json file
            file_path = os.path.join(results_root, pdb, file_name)
            score_df = pd.read_json(file_path, lines=True)
            score_df.rename(columns={'label': 'score'}, inplace=True)
            cols = ['id', 'score']
            for col in score_df.columns:
                if col not in cols:
                    del score_df[col]
            score_df['lig_id'] = score_df['id'].apply(lambda x: x.split('-')[1].split('_')[0])


            # select the highest-scoring entry per group
            score_df = score_df.groupby('lig_id').max()


            # add target, true binder, true L1 binder
            score_df['topactid'] = topact_id
            score_df['target'] = int(target_df[target_df['#T'] == pdb].target.iloc[0])
            score_df['label'] = score_df['id'].apply(lambda x: 1 if x.split('-')[1].split('_')[0] in act_id else 0)

            score_df_ls.append(score_df)

        combined_df = pd.concat(score_df_ls, axis=0)
        topnum1, SR1, topnum5, SR5, topnum10, SR10, EF1, EF5, EF10 = obtain_metircs(combined_df, k1=k1, k2=k2, k3=k3)

        # results
        print("\nSuccess Rate Across All PDBs:")
        print(f"Success Rate top-{int(k1 * 100)}(%): {SR1:.2f}%")
        print(f"Success Rate top-{int(k2 * 100)}(%): {SR5:.2f}%")
        print(f"Success Rate top-{int(k3 * 100)}(%): {SR10:.2f}%")
        print(f"Enrichment Factor top-{int(k1 * 100)}: {EF1:.2f}")
        print(f"Enrichment Factor top-{int(k2 * 100)}: {EF5:.2f}")
        print(f"Enrichment Factor top-{int(k3 * 100)}: {EF10:.2f}")
        print('-' * 30)

        d['ckpt'].append(os.path.splitext(os.path.basename(ckpt_path))[0])
        d[f'SR top-{int(k1 * 100)}(%)'].append(round(SR1, 2))
        d[f'SR top-{int(k2 * 100)}(%)'].append(round(SR5, 2))
        d[f'SR top-{int(k3 * 100)}(%)'].append(round(SR10, 2))

        d[f'EF top-{int(k1 * 100)}'].append(round(EF1, 2))
        d[f'EF top-{int(k2 * 100)}'].append(round(EF5, 2))
        d[f'EF top-{int(k3 * 100)}'].append(round(EF10, 2))

    results_df = pd.DataFrame(d)
    results_file = os.path.join(PROJ_DIR, args.results_dir, f'{version}_screening_results_{args.test_name}.csv')
    results_df.to_csv(results_file, index=False)

    
if __name__ == '__main__':
    main(parse())
