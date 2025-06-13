#!/bin/bash
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

#------------------------------CASF2016-screening power--------------------------- 
python screening_power_all.py \
    --test_dir /data1/baidu/data/BioScore_data/PLI \
    --dataset_info_file /data1/baidu/data/BioScore_data/PLI/PLI_screening_info.json \
    --dataset_name CASF2016 \
    --test_suffix logpp \
    --task PLI \
    --output_type docking/screening \
    --results_dir datasets/${Date_Type}/${Mission_Type}/${Model}/version_${VER} \
    --topk_start ${TOPK_START} \
    --topk_end ${TOPK_END} \
    --start 0 \
    --end 57 \
    --batch_size 128 \
    --num_workers 4 \
    --gpu 0

#------------------------------DEKOIS2.0-screening power--------------------------- 
python screening_power_all.py \
    --test_dir /data1/baidu/data/BioScore_data/PLI \
    --dataset_info_file /data1/baidu/data/BioScore_data/PLI/PLI_screening_info.json \
    --dataset_name DEKOIS2.0_SP \
    --test_suffix logpp \
    --family_name all \
    --task PLI \
    --output_type docking/screening \
    --results_dir datasets/${Date_Type}/${Mission_Type}/${Model}/version_${VER} \
    --topk_start ${TOPK_START} \
    --topk_end ${TOPK_END} \
    --start 0 \
    --end 81 \
    --batch_size 128 \
    --num_workers 4 \
    --gpu 0
    
# ------------------------------DUD-E-screening power--------------------------- 
python screening_power_all.py \
    --test_dir /data1/baidu/data/BioScore_data/PLI \
    --dataset_info_file /data1/baidu/data/BioScore_data/PLI/PLI_screening_info.json \
    --dataset_name DUD-E_SP \
    --test_suffix logpp \
    --family_name all \
    --task PLI \
    --output_type docking/screening \
    --results_dir datasets/${Date_Type}/${Mission_Type}/${Model}/version_${VER} \
    --topk_start ${TOPK_START} \
    --topk_end ${TOPK_END} \
    --start 0 \
    --end 102 \
    --batch_size 128 \
    --num_workers 4 \
    --gpu 0


#----------------------------PPIBenchmark-screening power-------------------------
# dataset_name: "PDBbind_PP”
python screening_power_all.py \
    --test_dir /data1/baidu/data/BioScore_data/PPI \
    --dataset_info_file /data1/baidu/data/BioScore_data/PPI/PPI_screening_info.json \
    --dataset_name PDBbind_PP \
    --test_suffix logpp \
    --family_name all \
    --task PPI \
    --output_type docking/screening \
    --results_dir datasets/${Date_Type}/${Mission_Type}/${Model}/version_${VER} \
    --topk_start ${TOPK_START} \
    --topk_end ${TOPK_END} \
    --start 0 \
    --end 79 \
    --batch_size 256 \
    --num_workers 4 \
    --gpu 0

#----------------------------peptide-MHC-screening power------------------------- 
python screening_power_all.py \
    --test_dir /data1/baidu/data/BioScore_data/PLI \
    --dataset_info_file /data1/baidu/data/BioScore_data/PPI/PPI_screening_info.json \
    --dataset_name pMHC \
    --test_suffix logpp_pli \
    --task PLI \
    --output_type docking/screening \
    --results_dir datasets/${Date_Type}/${Mission_Type}/${Model}/version_${VER} \
    --topk_start ${TOPK_START} \
    --topk_end ${TOPK_END} \
    --start 0 \
    --end 11 \
    --batch_size 128 \
    --num_workers 4 \
    --gpu 0