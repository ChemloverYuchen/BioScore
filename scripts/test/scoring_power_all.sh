#!/bin/bash
VER=0
Date_Type=PLI+PPI
Mission_Type=PLI_finetune
Model=bioscore
TOPK_START=0
TOPK_END=1

#---------------------CASF2016-scoring power-------------------------------
# Mission_Type=PLI_finetune
get infer results
python scoring_power_all.py \
    --test_set /data1/baidu/data/BioScore_data/PLI/CASF2016_scoring_10A.pkl \
    --task PLI\
    --output_type scoring/ranking \
    --results_dir datasets/${Date_Type}/${Mission_Type}/${Model}/version_${VER} \
    --test_name CASF2016_logpp \
    --topk_start ${TOPK_START} \
    --topk_end ${TOPK_END}\
    --batch_size 64 \
    --num_workers 4 \
    --gpu 0

# evaluate results
python evaluate_scoring_all.py \
    --results_dir datasets/${Date_Type}/${Mission_Type}/${Model}/version_${VER} \
    --test_name CASF2016_logpp \
    --topk_start ${TOPK_START} \
    --topk_end ${TOPK_END}\


# # ---------------------PPIBenchmark-scoring power------------------------
# # Mission_Type=PPI_finetune
# get infer results
# python scoring_power_all.py \
#     --test_set /data1/baidu/data/BioScore_data/PPI/PDBbind_PP_scoring.pkl \
#     --task PPI \
#     --output_type scoring/ranking \
#     --results_dir datasets/${Date_Type}/${Mission_Type}/${Model}/version_${VER} \
#     --test_name PPIbenchmark_logpp \
#     --topk_start ${TOPK_START} \
#     --topk_end ${TOPK_END}\
#     --batch_size 64 \
#     --num_workers 16 \
#     --gpu 0

# # evaluate results
# python evaluate_scoring_all.py \
#     --results_dir datasets/${Date_Type}/${Mission_Type}/${Model}/version_${VER} \
#     --test_name PPIbenchmark_logpp \
#     --topk_start ${TOPK_START} \
#     --topk_end ${TOPK_END}


# #---------------------CPSet-scoring power-------------------------------
# # Mission_Type=PLI_finetune
# # get infer results
# python scoring_power_all.py \
#     --test_set /data1/baidu/data/BioScore_data/PLI/CPSet_scoring_10A.pkl \
#     --task PLI\
#     --output_type scoring/ranking \
#     --results_dir datasets/${Date_Type}/${Mission_Type}/${Model}/version_${VER} \
#     --test_name CPSet_logpp \
#     --topk_start ${TOPK_START} \
#     --topk_end ${TOPK_END}\
#     --batch_size 64 \
#     --num_workers 4 \
#     --gpu 0

# # evaluate results
# python evaluate_scoring_all.py \
#     --results_dir datasets/${Date_Type}/${Mission_Type}/${Model}/version_${VER} \
#     --test_name CPSet_logpp \
#     --topk_start ${TOPK_START} \
#     --topk_end ${TOPK_END}


# # ---------------------MC_noCPSet-scoring power------------------------
# # Mission_Type=PLI_finetune
# # get infer results
# python scoring_power_all.py \
#     --test_set /data1/baidu/data/BioScore_data/PLI/MC_noCPSet_scoring_10A.pkl \
#     --task PLI\
#     --output_type scoring/ranking \
#     --results_dir datasets/${Date_Type}/${Mission_Type}/${Model}/version_${VER} \
#     --test_name MC_noCPSet_logpp \
#     --topk_start ${TOPK_START} \
#     --topk_end ${TOPK_END} \
#     --batch_size 64 \
#     --num_workers 4 \
#     --gpu 0

# # evaluate results
# python evaluate_scoring_all.py \
#     --results_dir datasets/${Date_Type}/${Mission_Type}/${Model}/version_${VER} \
#     --test_name MC_noCPSet_logpp \
#     --topk_start ${TOPK_START} \
#     --topk_end ${TOPK_END} \


# # ---------------------Carbohydrate-scoring power------------------------------
# # Mission_Type=PLI_finetune
# # get infer results
# python scoring_power_all.py \
#     --test_set /data1/baidu/data/BioScore_data/PLI/Carbohydrate_structure_10A.pkl \
#     --task PLI\
#     --output_type scoring/ranking \
#     --results_dir datasets/${Date_Type}/${Mission_Type}/${Model}/version_${VER} \
#     --test_name Carbohydrate_logpp \
#     --topk_start ${TOPK_START} \
#     --topk_end ${TOPK_END} \
#     --batch_size 64 \
#     --num_workers 4 \
#     --gpu 0

# # # evaluate results
# # python evaluate_scoring_all.py \
# #     --results_dir datasets/${Date_Type}/${Mission_Type}/${Model}/version_${VER} \
# #     --test_name Carbohydrate_logpp \
# #     --topk_start ${TOPK_START} \
# #     --topk_end ${TOPK_END} \



# # ---------------------SAbDab-scoring power------------------------
# # Mission_Type=PPI_finetune
# # get infer results
# python scoring_power_all.py \
#     --test_set /data1/baidu/data/BioScore_data/PPI/SAbDab_PP_processed/split0/test.pkl \
#     --task PPI \
#     --output_type scoring/ranking \
#     --results_dir datasets/${Date_Type}/${Mission_Type}/${Model}/version_${VER} \
#     --test_name SAbDab_test_logpp \
#     --topk_start ${TOPK_START} \
#     --topk_end ${TOPK_END}\
#     --batch_size 64 \
#     --num_workers 16 \
#     --gpu 0

# # evaluation of results of docking power by Spearman Rank Correlation
# python evaluate_scoring_all.py \
#     --results_dir  datasets/${Date_Type}/${Mission_Type}/${Model}/version_${VER} \
#     --test_name SAbDab_test_logpp \
#     --topk_start ${TOPK_START} \
#     --topk_end ${TOPK_END}