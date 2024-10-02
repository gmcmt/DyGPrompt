We provide the code (in pytorch) for our paper "DyGPrompt: Learning Feature and Time Prompts on Dynamic Graphs".

## Description

The repository is organised as follows:

*   `DYGPrompt/`: Contains all source code

*   `processed/`: You could download the datasets following the instructions in *Download Datasets* section, and place the downloaded datasets in this folder.

*   `downstream_data/`:Contains the last 20% of  Wikipedia, used for downstream training, validating and testing

*   `utils/`: Pre-process the dataset for pretrain and downstream task

*   `requirements.txt`: Listing the dependencies of the project.

## Download Dataset

Download the sample datasets (eg. wikipedia and reddit) from
[here](http://snap.stanford.edu/jodie/) and store their csv files in a folder named
`processed/`.

## Process the data

We use the dense `npy` format to save the features in binary format. If edge features or nodes
features are absent, they will be replaced by a vector of zeros.

```{bash}
python utils/preprocess_data.py --data wikipedia --bipartite
python downstream_process.py
```

## Running experiments

### Link Prediction

Default dataset is WIKIPEDIA. You need to change the corresponding parameters in pretrain\_origi.py and  downstream\_link\_fewshot.py to train and evaluate on other datasets.

Pretrain:

```{bash}
python pretrain_origi.py --use_memory --prefix TGN
python -u pretrain.py -d wikipedia --bs 200 --uniform  --n_degree 20 --agg_method attn --attn_mode prod --gpu 0 --n_head 2 --prefix TGAT
```

Prompt tune and test:

```{bash}
python downstream_link_fewshot.py --use_memory --prefix TGN
python -u downstream_link_meta.py -d wikipedia --bs 512 --uniform  --n_degree 20 --agg_method attn --attn_mode prod --gpu 0 --n_head 2 --prefix TGAT --name TGAT_LINK --fn saved
```

### Node Classification

Default dataset is WIKIPEDIA. You need to change the corresponding parameters in downstream\_meta.py to train and evaluate on other datasets.

Prompt tune and test:

```{bash}
    python downstream_meta.py --use_memory --prefix TGN
    python -u node_task_meta.py -d genre --bs 100 --uniform  --n_degree 20 --agg_method attn --attn_mode prod --gpu 0 --n_head 2 --prefix TGAT
```
