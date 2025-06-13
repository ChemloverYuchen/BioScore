#!/bin/bash

# Example (PLI)

# # output_type: scoring/ranking
# python ../inference.py \
#     --test_set ./data/BioScore_data/PLI_example_10A.pkl \
#     --task PLI \
#     --output_type scoring/ranking \
#     --ckpt ../datasets/PLI+PPI/PLI_finetune/bioscore/version_0/checkpoint/final.ckpt \
#     --save_path ./results/PLI_example_scoring_ranking_results.jsonl \
#     --batch_size 32 \
#     --num_workers 4 \
#     --gpu 0
    
# # # output_type: docking/screening
python ../inference.py \
    --test_set ./data/BioScore_data/PLI_example_10A.pkl \
    --task PLI \
    --output_type docking/screening \
    --ckpt ../datasets/PLI+PPI/PLI_finetune/bioscore/version_0/checkpoint/final.ckpt \
    --save_path ./results/PLI_example_docking_screening_results.jsonl \
    --batch_size 32 \
    --num_workers 4 \
    --gpu 0
