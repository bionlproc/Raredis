# Raredis

This repository contains code for our paper:  Comparison of pipeline, sequence-to-sequence, and generative language models for end-to-end relation extraction: experiments with the rare disease use-case.

# Dataset
The full modified dataset is available at this [link](https://drive.google.com/drive/folders/1XkfRKwWdrrV-wdzp9GdEXJHTHit9GbNi?usp=sharing).


## Seq2rel
### 1. Preparing the environment
Please follow the original seq2rel repo for installation and environment preparation guidelines [here](https://github.com/JohnGiorgi/seq2rel/blob/main/README.md)  

### 2. Prepare data
We follow the same linearization schema as the original author.  
Datasets are tab-separated files where each example is contained on its own line. The first column contains the text, and the second column contains the relations. Relations themselves must be serialized to strings.

```
SCAN1 has been identified in a single Saudi Arabian family. It has not been identified in other ataxic individuals. The diagnosis of SCAN1 is made on history and clinical signs as listed above. DNA testing for mutations in TDP1 is only available on a research basis.	SCAN1 @RAREDISEASE@ It @ANAPHOR@ @Anaphora@ 
```  
[Seq2rel/data_prep_REL.py](https://github.com/shashank140195/Raredis/tree/main/Seq2rel) will generate files in the desired format for seq2rel.The pre processed input files are present in [Seq2rel/preprocees_data](https://github.com/shashank140195/Raredis/tree/main/Seq2rel/preprocees_data) folder.

### 3. Model Training
We trained our model on Google Colab Pro+ using A100 GPU.  
Git clone the John Giorgi's seq2rel github repo in the desired location in your drive [Seq2rel repo](https://github.com/JohnGiorgi/seq2rel)  


### Training

To train the model, use the [`allennlp train`](https://docs.allennlp.org/main/api/commands/train/) command with [one of our configs](https://github.com/shashank140195/Raredis/tree/main/Seq2rel/config) (or write your own!)

For example, to train a model on the Raredis, first, preprocess the data mentioned in the previous step or directly use the already pre-processed data from [Seq2rel/preprocees_data](https://github.com/shashank140195/Raredis/tree/main/Seq2rel/preprocees_data) folder.

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
Our model train Google collab file here [link](https://colab.research.google.com/drive/1sShXyD-E9CnHZKzk7ZhJekqlRdPd6IqH?usp=sharing)  

### 4. Evaluation
For overall and per relation score, run [Seq2rel/eval_rel_type.py](https://github.com/shashank140195/Raredis/tree/main/Seq2rel). Make sure you change the path to the trained model and gold test file.


## BioGPT
All the Experiments are done in Google Colab Pro+ using A100 GPU.

### 1. Data Prep
1. First run the [BioGPT/scripts/data_preparation/rawToJSON.py](https://github.com/shashank140195/Raredis/tree/main/BioGPT/scripts/data_preparation) to convert the raw files in the JSON format. This script adds/removes the instruction to the input sequence and adds/removes entity type for the target sequence.  
2. Run [BioGPT/scripts/data_preparation/rel_is_preprocess.py](https://github.com/shashank140195/Raredis/tree/main/BioGPT/scripts/data_preparation) to pre-process the JSON data in rel-is input format. This will output .pmid, .x, and .y files for each split.   
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

The pre-processed data can also be found [here](https://github.com/shashank140195/Raredis/tree/main/BioGPT/data)

### Training  

### 1. Requirements and Installation
Git clone the BioGPT repo 
```  
!git clone https://github.com/microsoft/BioGPT.git
```  

and then follow the original GitHub repo to install the necessary libraries to work with BioGPT [here](https://github.com/microsoft/BioGPT) or run the following cell.
```  
!git clone https://github.com/pytorch/fairseq  
import os
os.chdir("/content/fairseq")
!git checkout v0.12.0
!pip install .
!python setup.py build_ext --inplace
```  
Moses
```
os.chdir("/content/BioGPT")
!git clone https://github.com/moses-smt/mosesdecoder.git
!export MOSES=${PWD}/mosesdecoder
```
FastBPE
```
!git clone https://github.com/glample/fastBPE.git
!export FASTBPE=${PWD}/fastBPE
os.chdir("fastBPE")
!g++ -std=c++11 -pthread -O3 fastBPE/main.cc -IfastBPE -o fast
```
Sacremoses
```
!pip install sacremoses
!pip install tensorboardX
```  
You can also follow our Google Colab working directory to follow the code for installation steps [here](https://colab.research.google.com/drive/1sMAbgWi-paABrweJO_fe5edz1r2uAEPZ?usp=sharing).

### 2. Model Download

1. The link to the pre-trained BioGPT and BioGPT large is provided on the original GitHub repo [here](https://github.com/microsoft/BioGPT). We observed that sometimes the URL doesn't work so alternatively you can use [this link to download BioGPT medium](https://drive.google.com/file/d/1niani8rR_Wgtu-62I0OXDPFW1izW_ZCw/view?usp=drive_link)(4GB) or [this link to download BioGPT large](https://drive.google.com/file/d/16r614gaXllWq9zJvK437zoHs9yMpztNl/view?usp=drive_link)(18GB) from our google drive.    
```
os.chdir("/content/BioGPT/")
os.mkdir("checkpoints")
os.chdir("checkpoints")
!wget https://msramllasc.blob.core.windows.net/modelrelease/BioGPT/checkpoints/Pre-trained-BioGPT.tgz
!tar -zxvf Pre-trained-BioGPT.tgz
```

if the above URL doesn't work, (sometimes public access error), try running the below code to copy the BioGPT model from your Google Drive to Google Collab
```
os.chdir("/content/BioGPT/")
os.mkdir("checkpoints")
os.chdir("checkpoints")
os.mkdir("Pre-trained-BioGPT")

# copy the model checkpoint from google drive
%cp -av "/content/drive/MyDrive/BioGPT/pre_trained_model_med/checkpoint.pt" "/content/BioGPT/checkpoints/Pre-trained-BioGPT"
```

The model path hierarchy should look like this:  
<img width="278" alt="Screenshot 2023-08-24 at 12 31 10 PM" src="https://github.com/shashank140195/Raredis/assets/69673535/0b407e6e-2485-43f9-bf76-f36408337cfa">


2. Create a folder name "Raredis" under the data subfolder in the BioGPT path and paste the [BioGPT/data/raw](https://github.com/shashank140195/Raredis/tree/main/BioGPT/data) folder inside it. 
``` 
os.chdir("/content/BioGPT/data")
os.mkdir("Raredis")
%cp -av "content/drive/Mydrive/raw" "/content/BioGPT/data/Raredis/"
```
File should look like this:  


# change the path to your actual raw path
%cp -av "content/drive/Mydrive/raw" "/content/BioGPT/data/Raredis/"

3. Copy the [Re-Raredis](https://github.com/shashank140195/Raredis/tree/main/BioGPT/RE-Raredis) under the subfolder "examples" in the BioGPT path.  

4. Run [preprocess.sh](https://github.com/shashank140195/Raredis/blob/main/BioGPT/RE-Raredis/preprocess.sh)  
```
!bash preprocess.sh
```  
The above command will create 1 more folder named "relis-bin" under the same folder as raw path.

5. Run train.sh to begin training the model
```
!bash train.sh
```  

6. After training run infer.sh. This script run inference on the test.txt and generates .detok file
```
!bash infer.sh
```  

7. Post processing  
After inference, run the [BioGPT/scripts/postprocess](https://github.com/shashank140195/Raredis/tree/main/BioGPT/scripts/postprocess) to fetch the inference in the desired json format.

