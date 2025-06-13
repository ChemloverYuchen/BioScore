#!/bin/bash
VER=0
Date_Type=PLI+PPI
Mission_Type=PLI_finetune
Model=bioscore
TOPK_START=0
TOPK_END=1

# ---------------------CASF2016-ranking power------------------------------- 
# get infer results（same as scoring, no need to rerun this step after scoring test!）
python scoring_power_all.py \
    --test_set /data1/baidu/data/BioScore_data/PLI/CASF2016_scoring_10A.pkl \
    --task PLI \
    --output_type scoring/ranking \
    --results_dir datasets/${Date_Type}/${Mission_Type}/${Model}/version_${VER} \
    --test_name CASF2016_logpp \
    --topk_start ${TOPK_START} \
    --topk_end ${TOPK_END} \
    --batch_size 64 \
    --num_workers 4 \
    --gpu 0

# evaluate results
python evaluate_ranking_all.py \
    --casf_root /data1/baidu/data/protein-mol/CASF-2016 \
    --results_dir datasets/${Date_Type}/${Mission_Type}/${Model}/version_${VER} \
    --test_name CASF2016_logpp \
    --topk_start ${TOPK_START} \
    --topk_end ${TOPK_END}

