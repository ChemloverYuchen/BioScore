#!/bin/bash

# Example (PLI)
# step1: extract pockets (only for PLI)
# python ../scripts/data_process/extract_pocket.py \
#     --src_root ./data/raw_data \
#     --tgt_root ./data/processed_data \
#     --interface_dist_th 10.0

# # step2: preprocess
python ../scripts/data_process/process_PLI_pocket.py \
    --benchmark_dir ./data \
    --out_dir ./data/BioScore_data \
    --json_name example_affinities \
    --outname PLI_example_10A \
    --interface_dist_th 10.0
    