#!/bin/bash

# Example (PLI)
# step1: extract pockets (only for PLI)
# python ../scripts/data_process/extract_pocket.py \
#     --src_root ./data/PLI/raw_data \
#     --tgt_root ./data/PLI/processed_data \
#     --interface_dist_th 10.0

# # step2: preprocess
# python ../scripts/data_process/process_PLI_pocket.py \
#     --benchmark_dir ./data/PLI \
#     --out_dir ./data/BioScore_data \
#     --json_name example_affinities \
#     --outname PLI_example_10A \
#     --interface_dist_th 10.0


# Example (PPI)
# step1: preprocess
python ../scripts/data_process/process_PPI.py \
    --index_file ./data/PPI/example_index.csv \
    --pdb_dir ./data/PPI/pdb_files \
    --out_dir ./data/BioScore_data \
    --outname PPI_example_6A \
    --interface_dist_th 6.0