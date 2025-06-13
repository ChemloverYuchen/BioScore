import os
import sys
import json
import argparse
import numpy as np
import pandas as pd

from scipy.stats import spearmanr, kendalltau, pearsonr
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, mean_absolute_error


PROJ_DIR = os.path.abspath(os.path.join(
    os.path.split(os.path.abspath(__file__))[0],
    '..', '..'
))
sys.path.append(PROJ_DIR)
print(PROJ_DIR)

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


def parse():
    parser = argparse.ArgumentParser(description='evaluate scoring power')
    parser.add_argument('--results_dir', type=str, required=True, help='Path to the results directory')
    parser.add_argument('--test_name', type=str, default='PDBbind', help='Test dataset')
    parser.add_argument('--dedup_file', type=str, default=None, help='json file of dedup ids')
    
    parser.add_argument('--topk_start', type=int, default=0, help='top-k ckpts for test: start idx')
    parser.add_argument('--topk_end', type=int, default=5, help='top-k ckpts for test: end idx')
 
    return parser.parse_args()


# Calculate Pearson & Spearman & Kendall's Tau
def obtain_metrics(df):
    #Calculate the Pearson correlation coefficient
    rp = pearsonr(df['gt'], df['label'])[0]
    
    label_rank = df['label'].rank(ascending=False)
    gt_rank = df['gt'].rank(ascending=False)
    rs = spearmanr(label_rank, gt_rank)[0]
    rk = kendalltau(label_rank, gt_rank)[0]
    
    regr = linear_model.LinearRegression()
    regr.fit(df['label'].values.reshape(-1,1), df['gt'].values.reshape(-1,1))
    preds = regr.predict(df['label'].values.reshape(-1,1))
    mse = mean_squared_error(df['gt'], preds)
    
    num = df.shape[0]
    sd = np.sqrt((mse * num) / (num - 1))
    
    # rmse = mean_squared_error(df['gt'], preds, squared=False)
    # mae = mean_absolute_error(df['gt'], preds)
    
    rmse = mean_squared_error(df['gt'], df['label'], squared=False)
    mae = mean_absolute_error(df['gt'], df['label'])
    
    print("Pearson correlation coefficient (Rp): %.4f"%rp)
    print("Spearman correlation coefficient (Rs): %.4f"%rs)
    print("Kendall correlation coefficient (Rk): %.4f"%rk)
    
    print("RMSE: %.4f"%rmse)
    print("MAE: %.4f"%mae)
                                         
    print("The regression equation: logKa = %.2f + %.2f * Score"%(float(regr.coef_[0].item()), float(regr.intercept_[0].item())))
    print("Number of favorable sample (N): %d"%num)
    print("Standard deviation in fitting (SD): %.2f"%sd)

    
    return rp, rs, rk, sd, num, rmse, mae

    
def main(args):
    version = args.results_dir.split('/')[-1]
    topk_map_path = os.path.join(PROJ_DIR, args.results_dir, 'checkpoint', 'topk_map.txt')
    
    with open(topk_map_path, 'r') as f:
        lines = f.readlines()
        
    # dedup
    if args.dedup_file is not None:
        with open(args.dedup_file, 'r') as f:
            dedup_dict = json.load(f)
        dedup_ids = set(dedup_dict.keys())
        test_suffix = '_dedup'
    else:
        test_suffix = ''
    
    d = {'ckpt': [], 'Pearson': [], 'Spearman': [], 'Kendall': [], 'RMSE': [], 'MAE': []}
        
    for line in lines[args.topk_start:args.topk_end]:
        metric, ckpt_path = line.strip('\n').split(': ')
        ckpt_path = os.path.join(PROJ_DIR, ckpt_path)
        file_name = os.path.splitext(os.path.basename(ckpt_path))[0] + '_results.jsonl'
        file_path = os.path.join(PROJ_DIR, args.results_dir, f'scoring_results_{args.test_name}', file_name)
        
        print(f'{file_name} scoring results:')
        
        # load json file
        data = pd.read_json(file_path, lines=True)
        print(len(data))
        
        # dedup
        if args.dedup_file is not None:
            data = data[data['id'].isin(dedup_ids)].copy()
            print(f'After dedup: {len(data)}')

        # calculate metrics
        rp, rs, rk, sd, num, rmse, mae = obtain_metrics(data)                          
        print('-' * 30)
        
        # results
        d['ckpt'].append(os.path.splitext(os.path.basename(ckpt_path))[0])
        d['Pearson'].append(round(rp, 4))
        d['Spearman'].append(round(rs, 4))
        d['Kendall'].append(round(rk, 4))
        d['RMSE'].append(round(rmse, 4))
        d['MAE'].append(round(mae, 4))

    results_df = pd.DataFrame(d)
    results_file = os.path.join(PROJ_DIR, args.results_dir, f'{version}_scoring_results_{args.test_name}{test_suffix}.csv')
    results_df.to_csv(results_file, index=False)

    
if __name__ == '__main__':
    main(parse())
