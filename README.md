# Raredis

This repository contains code for our paper:  Comparison of pipeline, sequence-to-sequence, and generative language models for end-to-end relation extraction: experiments with the rare disease use-case.

## Dataset
The full original dataset is availabe at this link: [Raredis Modified dataset](https://drive.google.com/drive/folders/1XkfRKwWdrrV-wdzp9GdEXJHTHit9GbNi?usp=sharing).


## Seq2rel
### 1. Prepare data
[Seq2rel/data_prep_REL.py](https://github.com/shashank140195/Raredis/tree/main/Seq2rel) will generate files in the desired format for seq2rel. The pre processed input files are present in [Seq2rel/data](https://github.com/shashank140195/Raredis/tree/main/Seq2rel/data) folder

### 2. Model Training
We trained our model on google Colab pro+ using A100 GPU.
a. Git clone the John Giorgi's seq2rel github repo in the desired location in your drive[Seq2rel repo](https://github.com/JohnGiorgi/seq2rel)

### 3. Evaluation
For overall and per relation score, run [Seq2rel/eval_rel_type.py](https://github.com/shashank140195/Raredis/tree/main/Seq2rel). Make sure you change the path to the trained model and gold test file.