# # #!/bin/bash
VER=0
Date_Type=PLI+PPI
Mission_Type=PLI+PPI_pretrain
Model=bioscore
TOPK_START=0
TOPK_END=1

# CASF2016: 57 targets
# DEKOIS2.0_SP: 81 targets
# DUD-E_SP: 102 targets

# PDBbind_PP：79 targets
# pMHC：11 targets 
       
# ------------------------------CASF2016-screening power--------------------------- 
# evaluate results
python evaluate_screening_casf.py \
    --casf_root /data1/baidu/data/protein-mol/CASF-2016/power_screening \
    --results_dir datasets/${Date_Type}/${Mission_Type}/${Model}/version_${VER} \
    --test_name CASF2016_logpp_all \
    --topk_start ${TOPK_START} \
    --topk_end ${TOPK_END} \
    --k1 0.01 \
    --k2 0.05 \
    --k3 0.10


# ------------------------------DEKOIS2.0-screening power---------------------------
# evaluate results
python evaluate_screening_all.py \
    --results_dir datasets/${Date_Type}/${Mission_Type}/${Model}/version_${VER} \
    --dataset_info_file /data1/baidu/data/BioScore_data/PLI/PLI_screening_info.json \
    --dataset_name DEKOIS2.0_SP \
    --test_suffix logpp \
    --family_name all \
    --topk_start ${TOPK_START} \
    --topk_end ${TOPK_END} \
    --alpha 80.5 \
    --k1 0.005 \
    --k2 0.01 \
    --k3 0.05

# ------------------------------DUD-E-screening power--------------------------- 
# evaluate results
python evaluate_screening_all.py \
    --results_dir datasets/${Date_Type}/${Mission_Type}/${Model}/version_${VER} \
    --dataset_info_file /data1/baidu/data/BioScore_data/PLI/PLI_screening_info.json \
    --dataset_name DUD-E_SP \
    --test_suffix logpp \
    --family_name all \
    --topk_start ${TOPK_START} \
    --topk_end ${TOPK_END} \
    --alpha 80.5 \
    --k1 0.005 \
    --k2 0.01 \
    --k3 0.05

# ----------------------------PPIBenchmark-screening power-------------------------
# (different from others!)
# evaluate results
python evaluate_screening_ppi.py \
    --results_dir datasets/${Date_Type}/${Mission_Type}/${Model}/version_${VER} \
    --dataset_info_file /data1/baidu/data/BioScore_data/PPI/PPI_screening_info.json \
    --dataset_name PDBbind_PP \
    --test_suffix logpp \
    --family_name all \
    --topk_start ${TOPK_START} \
    --topk_end ${TOPK_END} \
    --k1 0.01 \
    --k2 0.05 \
    --k3 0.10


# ！！(check before test：test_suffix: xxx_pli)
# ----------------------------peptide-MHC-screening power------------------------- 
# evaluate results
python evaluate_screening_all.py \
    --results_dir datasets/${Date_Type}/${Mission_Type}/${Model}/version_${VER} \
    --dataset_info_file /data1/baidu/data/BioScore_data/PPI/PPI_screening_info.json \
    --dataset_name pMHC \
    --test_suffix logpp_pli \
    --family_name all \
    --topk_start ${TOPK_START} \
    --topk_end ${TOPK_END} \
    --alpha 80.5 \
    --k1 0.005 \
    --k2 0.01 \
    --k3 0.05
