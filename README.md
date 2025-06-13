# BioScore

This repository contains the implementation code for our research paper, [BioScore: A Foundational Scoring Function For Generalized Biomolecular Complexes]()

## Introduction
Here, we present BioScore—the first foundational scoring function designed to assess binding phenomenon across a broad range of biomolecular complexes. BioScore introduces innovations along three axes. (1) Representation: It departs from traditional atom/block discretizations by introducing interface-masking encoding and distance-aware edge construction, capturing dual-scale atomic and block-level features. This approach enables coarse-grained yet expressive representations applicable across molecular classes of various structural complexity. (2) Scoring Methodology: BioScore proposes a new structural assessment score that incorporates a learned statistical potential (via a mixture of density network, MDN) and a newly defined interaction-edge-aware score. (3) Training strategy: A pretraining–fine-tuning workflow (prioritize large-scale structural learning at the pre-training stage) balances performance across heterogeneous tasks, enabling plug-and-play adaptation to specific molecular systems.

![GUE](figures/bioscore.png)

## 1 Installation

Set up the environment according to `environment.yml`
```
conda env create -f environment.yml
```

## 2 Usage

You can use the provided checkpoints or train your own BioScore to score any biological complexes as follows.

### 2.1 Model Checkpoints

Model checkpoints are provided in the [datasets](./datasets) directory. For example, the following are commonly used for different tasks:

- [PLI+PPI_pretrain](./datasets/PLI+PPI/PLI+PPI_pretrain) recommended for docking/screening tasks of PLI or PPI
- [PLI_finetune](./datasets/PLI+PPI/PLI_finetune) recommended for scoring/ranking tasks of PLI
- [PPI_finetune](./datasets/PLI+PPI/PPI_finetune) recommended for scoring/ranking tasks of PPI


### 2.2 Inference with BioScore

Refer to the scripts in [example](./example) for an example of how to use the trained model for prediction.

#### Step 1: Data preprocess

First, convert raw data files (e.g., `.sdf` or `.pdb`) into processed `.pkl` files for model input.

```
cd example
bash data_preprocess.sh
```

#### Step 2: Inference

Second, choose the appropriate output format according to your task (`scoring/ranking` or `docking/screening`) and run the following scripts. 

```
bash inference_example.sh
```

Finally, the results will be saved in a `.jsonl` file, for example:

```
{"id": "3prs", "label": 7.677018165588379, "task": "PLI", "gt": 7.82}
```
 
### 2.3 Train your own BioScore

Before training, please ensure that the data has been processed into `.pkl` format as above and placed them in the correct path. Preprocessing scripts for different biological complexes are provided in [scripts/data_process](./scripts/data_process) for reference.

```
# split into train/valid
bash bioscore_data.sh
```

Scripts for pretraining and fine-tuning are provided in [bioscore_train.sh](bioscore_train.sh).

```
# pretrain and fine-tune
bash bioscore_train.sh
```


## 3 Reproducibility

We provide source data and codes for reproducing the experiments of the paper *BioScore* as follows.

### 3.1 Training data

The raw data used during training were primarily sourced from [PDBbind database(version 2020)](http://www.pdbbind.org.cn), and were structurally optimized following [our previous work](https://pubs.rsc.org/en/content/articlehtml/2023/sc/d3sc02044d).

### 3.2 Training configs

All parameter settings for pretraining and fine-tuning are available in [scripts/exps/configs](./scripts/exps/configs). We provide the following configs:

- `mix-PLI-PPI-pretrain`
- `mix-PLI-finetune`
- `mix-PPI-finetune`

### 3.3 Evaluation

Scripts for evaluation are provided in [scripts/test](./scripts/test) and all evaluation datasets mentioned in the paper can be publicly accessed.

- `PPI Evaluation`
    - [PPI Benchmark](xxxxx) (It will be released to the public very soon)
    - [SAbDab](https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabdab/)
    - [pMHC](https://zenodo.org/records/8324920)
- `PLI Evaluation`
    - [CASF-2016](http://www.pdbbind.org.cn)
    - [DEKOIS 2.0](https://zenodo.org/records/6859325)
    - [DUD-E](https://zenodo.org/records/6859325)
- `PNI Evaluation`
    - [PNI test](http://www.pdbbind.org.cn)
- `NLI Evaluation`
    - [NLI test](http://www.pdbbind.org.cn)
- `Special Cases`
    - [CPSet](https://github.com/huifengzhao/CPSet)
    - [Protein-Non-Peptidic Macrocycle Test](http://www.pdbbind.org.cn)
    - [PCAPRED](https://biosig.lab.uq.edu.au/csm_carbohydrate/datasets)


## 4 Citation

If you find our paper and code useful, please consider citing the [preprint](xxxxx):

```
@article{xxxx,
  title={xxxx},
  author={xxxx},
  journal={xxxx},
  pages={xxxx},
  year={xxxx},
  publisher={xxxx}
}
```


## Contact
If you have any question, please feel free to email us (yuchenzhu@zju.edu.cn or jihongchen1005@gmail.com or lyt010312@163.com).