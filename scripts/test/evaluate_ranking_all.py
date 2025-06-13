import os
import sys
import argparse
import numpy as np
import pandas as pd

from scipy.stats import spearmanr, kendalltau, pearsonr
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

PROJ_DIR = os.path.abspath(os.path.join(
    os.path.split(os.path.abspath(__file__))[0],
    '..', '..'
))
sys.path.append(PROJ_DIR)
print(PROJ_DIR)

def parse():
    parser = argparse.ArgumentParser(description='evaluate ranking power')
    parser.add_argument('--casf_root', type=str, default='/data1/baidu/data/protein-mol/CASF-2016', help='Path to the CASF') 
                        
    parser.add_argument('--results_dir', type=str, required=True, help='Path to the results directory')
    parser.add_argument('--test_name', type=str, default='PDBbind', help='Test dataset')
    
    parser.add_argument('--topk_start', type=int, default=0, help='top-k ckpts for test: start idx')
    parser.add_argument('--topk_end', type=int, default=5, help='top-k ckpts for test: end idx')
 
    return parser.parse_args()


def cal_PI(score, logKa):
    """Define the Predictive Index function"""
    logKa, score = zip(*sorted(zip(logKa,score), key=lambda x:x[0], reverse=False))
    W=[]
    WC=[]
    n_ligands = len(logKa)
    for i in np.arange(0, n_ligands):
        for j in np.arange(i+1, n_ligands):
            w_ij=abs(logKa[i]-logKa[j])
            W.append(w_ij)
            if score[i] < score[j]:
                WC.append(w_ij)
            elif score[i] > score[j]:
                WC.append(-w_ij)
            else:
                WC.append(0)

    pi=float(sum(WC))/float(sum(W))
    return pi

    
def main(args):
    version = args.results_dir.split('/')[-1]
    topk_map_path = os.path.join(PROJ_DIR, args.results_dir, 'checkpoint', 'topk_map.txt')
    
    with open(topk_map_path, 'r') as f:
        lines = f.readlines()
    
    d = {'ckpt': [], 'Spearman': [], 'Kendall': [], 'PI': []}
        
    for line in lines[args.topk_start:args.topk_end]:
        metric, ckpt_path = line.strip('\n').split(': ')
        ckpt_path = os.path.join(PROJ_DIR, ckpt_path)
        file_name = os.path.splitext(os.path.basename(ckpt_path))[0] + '_results.jsonl'
        file_path = os.path.join(PROJ_DIR, args.results_dir, f'scoring_results_{args.test_name}', file_name)
        
        print(f'{file_name} scoring results:')
        
        # load json file
        core_file = os.path.join(args.casf_root, 'CoreSet.dat')                
        df = pd.read_csv(core_file, sep='[,,\t, ]+', header=0, engine='python')
        df_score = pd.read_json(file_path, lines=True)
        df_score.rename(columns={'id': '#code'}, inplace=True)                        
        print(len(df), len(df_score))
        
        testdf = pd.merge(df,df_score,on='#code', how='inner')
        df_groupby = testdf.groupby('target')
        
        # calculate Spearman/Kendall's Tau/PI
        spearman = df_groupby.apply(lambda x: x[["logKa","label"]].corr("spearman").iloc[1, 0]).mean()
        kendall = df_groupby.apply(lambda x: x[["logKa","label"]].corr("kendall").iloc[1, 0]).mean()
        PI = df_groupby.apply(lambda x: cal_PI(x.label, x.logKa)).mean()
        print('-' * 30)
        
        # results
        d['ckpt'].append(os.path.splitext(os.path.basename(ckpt_path))[0])
        d['Spearman'].append(round(spearman, 4))
        d['Kendall'].append(round(kendall, 4))
        d['PI'].append(round(PI, 4))
        
        print("The Spearman correlation coefficient (SP): %.4f"%spearman)
        print("The Kendall correlation coefficient (tau): %.4f"%kendall)
        print("The Predictive index (PI): %.4f"%PI)
    
    results_df = pd.DataFrame(d)
    results_file = os.path.join(PROJ_DIR, args.results_dir, f'{version}_ranking_results_{args.test_name}.csv')
    results_df.to_csv(results_file, index=False)

    
if __name__ == '__main__':
    main(parse())
