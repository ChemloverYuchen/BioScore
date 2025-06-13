#!/usr/bin/python
# -*- coding:utf-8 -*-
from collections import defaultdict
import re
import json
import pickle
from argparse import ArgumentParser
import os
import shutil

import numpy as np
import pandas as pd
def parse():
    parser = ArgumentParser(description='split train / valid / test')
    parser.add_argument('--data', type=str, required=True, help='Path to the binary data file')
    parser.add_argument('--out_dir', type=str, default=None, help='Directory to save results. Default the same as input data.')
    parser.add_argument('--valid_ratio', type=float, default=0.1,
                        help='Ratio of validation set')
    parser.add_argument('--test_ratio', type=float, default=0,
                        help='Ratio of test set')
    parser.add_argument('--seed', type=int, default=2023, help='seed')
    parser.add_argument('--benchmark', type=str, default=None, help='Path to benchmark json file. If this is enabled, '+ \
                        'Complexes from the benchmark will be used as test set and complexes from data will be used as train/valid.' + \
                        'Note that complexes sharing clusters with those in the benchmark will be dropped.')
    parser.add_argument('--seq_id', type=float, default=0.3, help='sequence identity threshold for clustering')
    return parser.parse_args()

def load_file(fpath):
    with open(fpath, 'rb') as fin:
        items = pickle.load(fin)
    return items

def save_file(items, fpath):
    with open(fpath, 'wb') as fout:
        pickle.dump(items, fout)

def exec_mmseq(cmd):
    r = os.popen(cmd)
    text = r.read()
    r.close()
    return text
def save_as_fasta(items, fasta_file):
    with open(fasta_file, 'w') as fout:
        for item in items:
            _id = item['id']
            # seq = item['rec_seqs'] + item['lig_seqs']
            seq = item['seq_protein1'] + item['seq_protein2']
            fout.write(f'>{_id}\n{seq}\n')

def main(args):
    np.random.seed(args.seed)

    items = load_file(args.data)
    # index = list(range(len(items)))
    # np.random.shuffle(index)
    
    # test set
    if args.benchmark is not None:
        benchmark = load_file(args.benchmark)
        benchmark_items = benchmark  # for second filtering
        print(f'Specified benchmark enabled as test set. Number of entries: {len(benchmark)}')
        test_index = list(range(len(items), len(items) + len(benchmark)))
        is_benchmark = [False for _ in items]
        items.extend(benchmark)
        is_benchmark.extend([True for _ in benchmark])

    # transfer to fasta format for clustering
    tmp_dir = './tmp'
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    else:
        raise ValueError(f'Working directory {tmp_dir} exists!')
    fasta = os.path.join(tmp_dir, 'seq.fasta')
    with open(fasta, 'w') as fout:
        for i, item in enumerate(items):
            _id = str(i)
            # seq = item['seq_protein1'] + item['seq_protein2']
            seq = item['rec_seqs'] + item['lig_seqs']
            fout.write(f'>{_id}\n{seq}\n')

    # mmseqs create database
    db = os.path.join(tmp_dir, 'DB')
    cmd = f'mmseqs createdb {fasta} {db}'
    exec_mmseq(cmd)
    db_clustered = os.path.join(tmp_dir, 'DB_clu')
    cmd = f'mmseqs cluster {db} {db_clustered} {tmp_dir} --min-seq-id {args.seq_id} -c 0.3'  # simlarity > 0.3 in the same cluster(coverage=0.3)
    res = exec_mmseq(cmd)
    num_clusters = re.findall(r'Number of clusters: (\d+)', res)
    if len(num_clusters):
        print(f'Number of clusters: {num_clusters[0]}')
    else:
        raise ValueError('cluster failed!')
    tsv = os.path.join(tmp_dir, 'DB_clu.tsv')
    cmd = f'mmseqs createtsv {db} {db} {db_clustered} {tsv}'
    exec_mmseq(cmd)
    
    # read tsv of class \t pdb
    with open(tsv, 'r') as fin:
        entries = fin.read().strip().split('\n')
    id2clu, clu2idx = {}, defaultdict(list)
    for entry in entries:
        cluster, _id = entry.strip().split('\t')
        id2clu[_id] = cluster
    for i, item in enumerate(items):
        cluster = id2clu[str(i)]
        clu2idx[cluster].append(i)

    clu_cnt = [len(clu2idx[clu]) for clu in clu2idx]
    print(f'cluster number: {len(clu2idx)}, member number ' +
          f'mean: {np.mean(clu_cnt)}, min: {min(clu_cnt)}, ' +
          f'max: {max(clu_cnt)}')
    
    shutil.rmtree(tmp_dir)


    if args.out_dir is None:
        if args.benchmark is None:
            data_dir = os.path.split(args.data)[0]
        else:
            data_dir = os.path.split(args.benchmark)[0]
    else:
        data_dir = args.out_dir
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

    suffix = os.path.splitext(args.data)[-1]

    if args.benchmark is not None:
        fnames = ['train', 'valid']
        benchmark_clusters, other_clusters = [], []
        for c in clu2idx:
            in_test = False
            for i in clu2idx[c]:
                if is_benchmark[i]:
                    in_test = True
                    break
            if in_test:
                benchmark_clusters.append(c)
            else:
                other_clusters.append(c)
        np.random.shuffle(other_clusters)
        
        
        # transfer to fasta format for filtering
        tmp_dir = './tmp'
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)
        else:
            raise ValueError(f'Working directory {tmp_dir} exists!')

        benchmark_fasta = os.path.join(tmp_dir, f'benchmark_seq.fasta')
        others_fasta = os.path.join(tmp_dir, f'others_seq.fasta')

        all_other_items = []
        for c in other_clusters:
            for i in clu2idx[c]:
                items[i]['cluster'] = c
                all_other_items.append(items[i])
        print(f'Before filtering: others_items {len(all_other_items)}')
        
        save_as_fasta(benchmark_items, benchmark_fasta)
        save_as_fasta(all_other_items, others_fasta)
        
        # mmseqs create database + mmseq2 search
        test_db = os.path.join(tmp_dir, 'test_db')
        train_db = os.path.join(tmp_dir, 'train_db')
        result_db = os.path.join(tmp_dir, 'result_db')
        result_file = os.path.join(tmp_dir, 'result.tsv')
        cmd = (
            f'mmseqs createdb {benchmark_fasta} {test_db};'
            f'mmseqs createdb {others_fasta} {train_db};'
            f'mmseqs search {test_db} {train_db} {result_db} tmp --min-seq-id 0.3 -c 0.3;'
            f'mmseqs convertalis {test_db} {train_db} {result_db} {result_file};'
            
        )
        exec_mmseq(cmd)
        
        col_names = ['Query', 'Target', 'pident', 'alnlen', 'mismatch', 'gapopen', 'qstart', 'qend', 'tstart', 'tend', 'evalues', 'bits']
        result_df = pd.read_csv(result_file, sep='\t', header=None, names=col_names)
        similar_ids = set(list(result_df['Target'].unique()))
        print(f'Similar ids: {len(similar_ids)}')
        
        shutil.rmtree(tmp_dir)
        
        # split train/valid       
        valid_len = int(len(other_clusters) * args.valid_ratio)
        valid_clusters = other_clusters[-valid_len:]
        train_clusters = other_clusters[:-valid_len]
        for f, clusters in zip(fnames, [train_clusters, valid_clusters]):
            cur_items, cnt = [], 0
            f = os.path.join(data_dir, f + suffix)
            for c in clusters:
                for i in clu2idx[c]:
                    items[i]['cluster'] = c
                    # filtering by similar ids
                    if items[i]['id'] not in similar_ids:
                        cur_items.append(items[i])
                        cnt += 1
            save_file(cur_items, f)
            print('After filtering:')
            print(f'Save {len(clusters)} clusters, {cnt} entries to {f}')
            
    else:
        fnames = ['train', 'valid', 'test']
        clusters = list(clu2idx.keys())
        np.random.shuffle(clusters)
        valid_len, test_len = len(clu2idx) * args.valid_ratio, len(clu2idx) * args.test_ratio
        valid_len, test_len = int(valid_len), int(test_len)
        lengths = [len(clu2idx) - valid_len - test_len, valid_len, test_len]

        
        # split by clusters
        items_dict = {}
        start = 0
        for n, l in zip(fnames, lengths):
            assert 0 <= l and l < len(clusters)
            if l == 0:
                continue
            items_dict[n] = []
            end = start + l
            for c in clusters[start:end]:
                for i in clu2idx[c]:
                    items[i]['cluster'] = c
                    items_dict[n].append(items[i])
            start = end
        
        # second filtering
        # transfer to fasta format for filtering
        tmp_dir = './tmp'
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)
        else:
            raise ValueError(f'Working directory {tmp_dir} exists!')

        test_fasta = os.path.join(tmp_dir, f'test_seq.fasta')
        others_fasta = os.path.join(tmp_dir, f'others_seq.fasta')
        
        test_items, train_items, valid_items = items_dict['test'], items_dict['train'], items_dict['valid']
        others_items = train_items + valid_items
        print(f'Before filtering: train {len(train_items)}, valid {len(valid_items)}, test {len(test_items)}')
        
        save_as_fasta(test_items, test_fasta)
        save_as_fasta(others_items, others_fasta)
        
        # mmseqs create database + mmseq2 search
        test_db = os.path.join(tmp_dir, 'test_db')
        train_db = os.path.join(tmp_dir, 'train_db')
        result_db = os.path.join(tmp_dir, 'result_db')
        result_file = os.path.join(tmp_dir, 'result.tsv')
        cmd = (
            f'mmseqs createdb {test_fasta} {test_db};'
            f'mmseqs createdb {others_fasta} {train_db};'
            f'mmseqs search {test_db} {train_db} {result_db} tmp --min-seq-id 0.3 -c 0.3;'
            f'mmseqs convertalis {test_db} {train_db} {result_db} {result_file};'
            
        )
        exec_mmseq(cmd)
        
        col_names = ['Query', 'Target', 'pident', 'alnlen', 'mismatch', 'gapopen', 'qstart', 'qend', 'tstart', 'tend', 'evalues', 'bits']
        result_df = pd.read_csv(result_file, sep='\t', header=None, names=col_names)
        similar_ids = set(list(result_df['Target'].unique()))
        print(f'Similar ids: {len(similar_ids)}')
        
        shutil.rmtree(tmp_dir)
        
        print('After filtering:')
        # save to files
        for n in fnames:
            file_name = os.path.join(data_dir, n + suffix)
            before_items = items_dict[n]
            cnt = 0
            cluster_set = set()
            after_items = []
            for item in before_items:
                if item['id'] not in similar_ids:
                    after_items.append(item)
                    cluster_set.add(item['cluster'])
                    cnt += 1
            save_file(after_items, file_name)
            print(f'Save {len(cluster_set)} clusters, {cnt} entries to {file_name}')
            
if __name__ == '__main__':
    main(parse())
