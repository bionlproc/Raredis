# Raredis

This repository contains code for our paper:  Comparison of pipeline, sequence-to-sequence, and generative language models for end-to-end relation extraction: experiments with the rare disease use-case.

## Dataset
The full original dataset is availabe at this link: [Raredis Modified dataset](https://drive.google.com/drive/folders/1XkfRKwWdrrV-wdzp9GdEXJHTHit9GbNi?usp=sharing).


## Seq2rel
### 1. Preparing the environment
Please follow original seq2rel repo for installing guidelines here [Seq2rel repo](https://github.com/JohnGiorgi/seq2rel/blob/main/README.md)  

### 2. Prepare data
We follow the same linearization schema as original author.  
Datasets are tab-separated files where each example is contained on its own line. The first column contains the text, and the second column contains the relations. Relations themselves must be serialized to strings.

```
SCAN1 has been identified in a single Saudi Arabian family. It has not been identified in other ataxic individuals. The diagnosis of SCAN1 is made on history and clinical signs as listed above. DNA testing for mutations in TDP1 is only available on a research basis.	SCAN1 @RAREDISEASE@ It @ANAPHOR@ @Anaphora@ 
```  
[Seq2rel/data_prep_REL.py](https://github.com/shashank140195/Raredis/tree/main/Seq2rel) will generate files in the desired format for seq2rel. Move all the .txt files and .ann files of each split in different folders.The pre processed input files are present in [Seq2rel/data](https://github.com/shashank140195/Raredis/tree/main/Seq2rel/data) folder.

### 3. Model Training
We trained our model on google Colab pro+ using A100 GPU.  
a. Git clone the John Giorgi's seq2rel github repo in the desired location in your drive [Seq2rel repo](https://github.com/JohnGiorgi/seq2rel)  
b. 

### 4. Evaluation
For overall and per relation score, run [Seq2rel/eval_rel_type.py](https://github.com/shashank140195/Raredis/tree/main/Seq2rel). Make sure you change the path to the trained model and gold test file.