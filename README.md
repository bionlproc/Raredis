# Raredis

This repository contains code for our paper:  Comparison of pipeline, sequence-to-sequence, and generative language models for end-to-end relation extraction: experiments with the rare disease use-case.

## Dataset
The full original dataset is availabe at this link: [Raredis Modified dataset](https://drive.google.com/drive/folders/1XkfRKwWdrrV-wdzp9GdEXJHTHit9GbNi?usp=sharing).


## Seq2rel
### 1. Prepare data
Seq2rel/data_prep_REL.py will generate files in the desired format for seq2rel. The input files are present in Seq2rel/data folder

### 2. Model Training


### 3. Evaluation
For overall and per relation score, run Seq2rel/eval_rel_type.py. Make sure you change the path to the trained model and gold test file.