# Automatic Speech Recognition (ASR) with PyTorch

<p align="center">
  <a href="#about">About</a> •
  <a href="#results">Results</a> •
  <a href="#installation">Installation</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#demo">Demo</a> •
  <a href="#credits">Credits</a> •
  <a href="#license">License</a>
</p>

## About

This repository contains [Homework 1 of HSE DLA course](https://github.com/markovka17/dla/tree/2025/hw1_asr). \
Homework was solved using [Conformer](https://arxiv.org/abs/2005.08100) based model, in particular it's small version. Model was trained on train-clean-100 partition of Librespeech Dataset, CometML report of training - [report](https://www.comet.com/7embl4/asr/kxhxqnlx3j5wfg88knvolztgfuhxt6s6?&prevPath=%2F7embl4%2Fasr%2Fview%2Fnew%2Fpanels).

## Results
Metric         | test-clean | test-other
---------------|------------|-------------
Argmax WER     |    40.04   |    67.2    
Argmax CER     |    13.46   |    30.73
BeamSearch WER |    39.57   |    65.03
BeamSearch CER |    12.67   |    29.09

## Installation

Follow these steps to install the project:

0. (Optional) Create and activate new environment using [`conda`](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html) or `venv` ([`+pyenv`](https://github.com/pyenv/pyenv)).

   a. `conda` version:

   ```bash
   # create env
   conda create -n project_env python=PYTHON_VERSION

   # activate env
   conda activate project_env
   ```

   b. `venv` (`+pyenv`) version:

   ```bash
   # create env
   ~/.pyenv/versions/PYTHON_VERSION/bin/python3 -m venv project_env

   # alternatively, using default python version
   python3 -m venv project_env

   # activate env
   source project_env/bin/activate
   ```

1. Install all required packages

   ```bash
   pip install -r requirements.txt
   ```

2. Install `pre-commit`:
   ```bash
   pre-commit install
   ```

## How To Use

To train a model, run the following command:

```bash
python3 train.py -cn=CONFIG_NAME HYDRA_CONFIG_ARGUMENTS
```

Where `CONFIG_NAME` is a config from `src/configs` and `HYDRA_CONFIG_ARGUMENTS` are optional arguments.

To run inference (evaluate the model or save predictions):

```bash
python3 inference.py HYDRA_CONFIG_ARGUMENTS
```

## Demo

You can also see [Demo notebook](https://github.com/7embl4/asr/blob/main/notebooks/demo.ipynb) with full installation and usage processes

## Credits

This repository is based on a [PyTorch Project Template](https://github.com/Blinorot/pytorch_project_template).

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)
