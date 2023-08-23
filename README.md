# Raredis

This repository contains code for our paper:  Comparison of pipeline, sequence-to-sequence, and generative language models for end-to-end relation extraction: experiments with the rare disease use-case.

## Dataset
The full original dataset is availabe at this link: [Raredis Modified dataset](https://drive.google.com/drive/folders/1XkfRKwWdrrV-wdzp9GdEXJHTHit9GbNi?usp=sharing).


## Seq2rel
### 1. Preparing the environment
Please follow original seq2rel repo for installing guidelines.  
This repository requires Python 3.8 or later.

### Setting up a virtual environment

Before installing, you should create and activate a Python virtual environment. If you need pointers on setting up a virtual environment, please see the [AllenNLP install instructions](https://github.com/allenai/allennlp#setting-up-a-virtual-environment).

### Installing the library and dependencies

If you _do not_ plan on modifying the source code, install from `git` using `pip`

```bash
pip install git+https://github.com/JohnGiorgi/seq2rel.git
```

Otherwise, clone the repository and install from source using [Poetry](https://python-poetry.org/):

```bash
# Install poetry for your system: https://python-poetry.org/docs/#installation
# E.g. for Linux, macOS, Windows (WSL)
curl -sSL https://install.python-poetry.org | python3 -

# Clone and move into the repo
git clone https://github.com/JohnGiorgi/seq2rel
cd seq2rel

# Install the package with poetry
poetry install
```
### 2. Prepare data
[Seq2rel/data_prep_REL.py](https://github.com/shashank140195/Raredis/tree/main/Seq2rel) will generate files in the desired format for seq2rel. The pre processed input files are present in [Seq2rel/data](https://github.com/shashank140195/Raredis/tree/main/Seq2rel/data) folder

### 3. Model Training
We trained our model on google Colab pro+ using A100 GPU.  
a. Git clone the John Giorgi's seq2rel github repo in the desired location in your drive [Seq2rel repo](https://github.com/JohnGiorgi/seq2rel)  
b. 

### 4. Evaluation
For overall and per relation score, run [Seq2rel/eval_rel_type.py](https://github.com/shashank140195/Raredis/tree/main/Seq2rel). Make sure you change the path to the trained model and gold test file.