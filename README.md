# Raredis

This repository contains code for our paper:  Comparison of pipeline, sequence-to-sequence, and generative language models for end-to-end relation extraction: experiments with the rare disease use-case.

# Dataset
The full modified dataset is availabe at this [link](https://drive.google.com/drive/folders/1XkfRKwWdrrV-wdzp9GdEXJHTHit9GbNi?usp=sharing).


## Seq2rel
### 1. Preparing the environment
Please follow original seq2rel repo for installation and envioment preparation guidelines [here](https://github.com/JohnGiorgi/seq2rel/blob/main/README.md)  

### 2. Prepare data
We follow the same linearization schema as original author.  
Datasets are tab-separated files where each example is contained on its own line. The first column contains the text, and the second column contains the relations. Relations themselves must be serialized to strings.

```
SCAN1 has been identified in a single Saudi Arabian family. It has not been identified in other ataxic individuals. The diagnosis of SCAN1 is made on history and clinical signs as listed above. DNA testing for mutations in TDP1 is only available on a research basis.	SCAN1 @RAREDISEASE@ It @ANAPHOR@ @Anaphora@ 
```  
[Seq2rel/data_prep_REL.py](https://github.com/shashank140195/Raredis/tree/main/Seq2rel) will generate files in the desired format for seq2rel.The pre processed input files are present in [Seq2rel/preprocees_data](https://github.com/shashank140195/Raredis/tree/main/Seq2rel/preprocees_data) folder.

### 3. Model Training
We trained our model on google Colab pro+ using A100 GPU.  
Git clone the John Giorgi's seq2rel github repo in the desired location in your drive [Seq2rel repo](https://github.com/JohnGiorgi/seq2rel)  


### Training

To train the model, use the [`allennlp train`](https://docs.allennlp.org/main/api/commands/train/) command with [one of our configs](https://github.com/shashank140195/Raredis/tree/main/Seq2rel/config) (or write your own!)

For example, to train a model on the Raredis, first, preprocess the data mentioned in previous step or directly use the already pre processed data from [Seq2rel/preprocees_data](https://github.com/shashank140195/Raredis/tree/main/Seq2rel/preprocees_data) folder.

Then, call `allennlp train` with the [Raredis config we have provided](https://github.com/shashank140195/Raredis/blob/main/Seq2rel/config/raredis_bertlarge_config.jsonnet)

```bash
train_data_path="path/to/preprocessed/raredis/train.txt" \
valid_data_path="path/to/preprocessed/raredis/valid.txt" \
dataset_size=600 \
allennlp train "training_config/raredis.jsonnet" \
    --serialization-dir "output" \
    --include-package "seq2rel" 
```

The best model checkpoint (measured by micro-F1 score on the validation set), vocabulary, configuration, and log files will be saved to `--serialization-dir`. This can be changed to any directory you like. You can also follow
our model train google colab file here [link](https://colab.research.google.com/drive/1sShXyD-E9CnHZKzk7ZhJekqlRdPd6IqH?usp=sharing)  

### 4. Evaluation
For overall and per relation score, run [Seq2rel/eval_rel_type.py](https://github.com/shashank140195/Raredis/tree/main/Seq2rel). Make sure you change the path to the trained model and gold test file.


## BioGPT
### 1. Requirements and Installation

Please follow the original github repo to install the necessary libraries to work with BioGPT [here](https://github.com/microsoft/BioGPT).  
You can also follow our google colab working directory to follow the code for installation steps.

### 2. Data Prep
1. First run the [BioGPT/scripts/data_preparation/rawToJSON.py](https://github.com/shashank140195/Raredis/tree/main/BioGPT/scripts/data_preparation) to convert the raw files in the JSON format. This script adds/removes the instruction to the input sequence and add/removes entity type for the target sequence.  
2. Run [BioGPT/scripts/data_preparation/rel_is_preprocess.py](https://github.com/shashank140195/Raredis/tree/main/BioGPT/scripts/data_preparation) to pre process the JSON data in rel-is input format. This will output .pmid, .x and .y files for each split.   
split.pmid: It contains the document name  
split.x: It contains the input string  
split.y: it contains the target string  

For example. 
Original text
```
the incidence and prevalence of tarsal tunnel syndrome is unknown. the disorder is believed to affect males and females in equal numbers.
```  

Using rel_is_preprocess.py with enabling copy instruct and enabling ent type for the entities will generate  

split.pmid that contains
```
Tarsal-Tunnel-Syndrome
```  
 
split.x that contains
```
consider the abstract: $ the incidence and prevalence of tarsal tunnel syndrome is unknown. the disorder is believed to affect males and females in equal numbers. $ from the given abstract, find all the entities and relations among them. do not generate any token outside the abstract.
```  
split.y that contains
```
the relationship between raredisease tarsal tunnel syndrome and anaphor "the disorder" is antecedent.
```

The pre processed data can also be found [here](https://github.com/shashank140195/Raredis/tree/main/BioGPT/data)
