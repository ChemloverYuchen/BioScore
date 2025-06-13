#!/bin/bash
VER=0
Date_Type=PLI+PPI
Mission_Type=PLI+PPI_pretrain
Model=bioscore
TOPK_START=0
TOPK_END=1

# ---------------------CASF2016-docking power------------------------------
# get infer results
python docking_power_all.py \
    --test_set /data1/baidu/data/BioScore_data/PLI/CASF2016_docking_10A.pkl \
    --task PLI \
    --results_dir datasets/${Date_Type}/${Mission_Type}/${Model}/version_${VER} \
    --test_name CASF2016-logpp \
    --criterion rmsd \
    --topk_start ${TOPK_START} \
    --topk_end ${TOPK_END} \
    --batch_size 128 \
    --num_workers 4 \
    --output_type docking/screening \
    --gpu 0
    

# evaluate results
python evaluate_docking_pli.py \
    --results_dir datasets/${Date_Type}/${Mission_Type}/${Model}/version_${VER} \
    --test_name CASF2016-logpp \
    --topk_start ${TOPK_START} \
    --topk_end ${TOPK_END} \
    --k1 1 \
    --k2 2 \
    --k3 5


# ---------------------PPIBenchmark-docking power------------------------ 
# prepare dataset pkl and get infer results
python docking_power_all.py \
    --test_set /data1/baidu/data/BioScore_data/PPI/PDBbind_PP_docking.pkl \
    --task PPI \
    --results_dir datasets/${Date_Type}/${Mission_Type}/${Model}/version_${VER} \
    --test_name PPIbenchmark_logpp \
    --criterion dockq \
    --topk_start ${TOPK_START} \
    --topk_end ${TOPK_END} \
    --batch_size 64 \
    --num_workers 4 \
    --output_type docking/screening \
    --gpu 0

# evaluation of results of docking power by Spearman Rank Correlation
python evaluate_docking_ppi.py \
    --results_dir datasets/${Date_Type}/${Mission_Type}/${Model}/version_${VER} \
    --topk_start ${TOPK_START} \
    --topk_end ${TOPK_END} \
    --dockq 0.8 \
    --test_name PPIbenchmark_logpp