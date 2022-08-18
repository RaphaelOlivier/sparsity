# Adversarial sparsity

This code implements the **adversarial sparsity** metric introduced in the paper : [How many perturbations break this model? Evaluating robustness beyond
adversarial accuracy](https://arxiv.org/abs/2207.04129).

In order to run this code you must have installed RobustBench (https://github.com/RobustBench/robustbench), and Robustness (https://github.com/MadryLab/robustness). To run all experiments you should also have downloaded the DiffJPEG repository (https://github.com/mlomnitz/DiffJPEG) and the generated CIFAR10 data in https://github.com/deepmind/deepmind-research/tree/master/adversarial_robustness. Please specify the local paths to these repository and data, and other datasets, in `paths.py`

The scripts to run for a large part of our results are in the `run.sh` file. To run custom experiments, more details on argument options can be found in the `scripts/scripts_utils.py` file.
