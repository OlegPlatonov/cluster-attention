# Cluster Attention

This is the repository for the paper "Cluster Attention for Graph Machine Learning".

This repository can be used to run experiments with GNNs on many graph datasets including those from the recent GraphLand benchmark. Different graph machine learning models can be used, including MPNNs (GCN, GraphSAGE, GAT, LGT), GGT, and their CLATT-augmented versions.

How to use it:

If you want to use datasets from the GraphLand benchmark, download them from [Zenodo](https://zenodo.org/records/16895532) or [Kaggle](https://www.kaggle.com/datasets/bazhenovgleb/graphland) and put them in the `data` directory. If you want to use other supported datasets, they can be downloaded by the `Dataset` class automatically.

Install the dependencies from `environment.yaml` and run `main.py`.

Executing `main.py` runs a single experiment which might include hyperparameter search and multiple runs with the best hyperparameters to compute the mean and standard deviation of model performance. `main.py` can accept a number of arguments, see `get_args` function from `args.py` for a full list. A simple example is:

```
python main.py --name example_experiment --dataset hm-categories --split RL --transductive True --model GT --lr 3e-4 --dropout 0.1 --device cuda:0
```

Many arguments can accept a list of values. Passing a list of values rather than a single value will trigger grid search over these values. For example, to run grid search over learning rate and dropout probability, run the following command:

```
python main.py --name example_experiment_with_hparam_search --dataset hm-categories --split RL --transductive True --model GT --lr 3e-5 1e-4 3e-4 1e-3 3e-3 --dropout 0 0.1 0.2 --device cuda:0
```

The directory `scripts` contains commands to reproduce all the experiments from the paper. Specifically, there is one file for each dataset. Each line in each file runs a single experiment (with a single model and a single dataset) including hyperparameter search. In each file, there is one line for each of the models considered in the paper: GCN, GCN-CLATT, GraphSAGE, GraphSAGE-CLATT, LGT, LGT-CLATT, GGT-DW, GGT-DW-CLATT, GGT-Lap, GGT-Lap-CLATT. Note that running all the experiments takes considerable time (multiple days on a single A100 GPU).
