# Raredis

This repository contains code for our paper:  Comparison of pipeline, sequence-to-sequence, and generative language models for end-to-end relation extraction: experiments with the rare disease use-case.

# Dataset
The full modified dataset is available at this [link](https://drive.google.com/drive/folders/1XkfRKwWdrrV-wdzp9GdEXJHTHit9GbNi?usp=sharing).


## Seq2rel
All the Experiments are done in Google Colab Pro+ using A100 GPU.

### 1. Preparing the environment
Please follow the original seq2rel repo for installation and environment preparation guidelines [here](https://github.com/JohnGiorgi/seq2rel/blob/main/README.md)
Alternatively, run:  
```
pip install git+https://github.com/JohnGiorgi/seq2rel.git
```  

### 2. Prepare data
We follow the same linearization schema as provided by the authors.  
Datasets are tab-separated files where each example is contained on its own line. The first column contains the text, and the second column contains the relations. Relations themselves must be serialized to strings.

```
SCAN1 has been identified in a single Saudi Arabian family. It has not been identified in other ataxic individuals. The diagnosis of SCAN1 is made on history and clinical signs as listed above. DNA testing for mutations in TDP1 is only available on a research basis.	SCAN1 @RAREDISEASE@ It @ANAPHOR@ @Anaphora@ 
```  
[Seq2rel/data_prep_REL.py](https://github.com/shashank140195/Raredis/tree/main/Seq2rel) will generate files in the desired format for seq2rel. The pre processed input files are present in [Seq2rel/preprocees_data](https://github.com/shashank140195/Raredis/tree/main/Seq2rel/preprocees_data) folder.

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
1. First run the [BioGPT/scripts/data_preparation/rawToJSON.py](https://github.com/shashank140195/Raredis/tree/main/BioGPT/scripts/data_preparation) to convert the original files in the JSON format. This script adds/removes the instruction to the input sequence and adds/removes entity type for the target sequence.  
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

1. The link to the pre-trained BioGPT and BioGPT large is provided on the original GitHub repo [here](https://github.com/microsoft/BioGPT). We observed that sometimes the URL doesn't work so alternatively you can use [this link to download BioGPT medium](https://drive.google.com/file/d/1niani8rR_Wgtu-62I0OXDPFW1izW_ZCw/view?usp=drive_link)(4GB) or [this link to download BioGPT large](https://drive.google.com/file/d/16r614gaXllWq9zJvK437zoHs9yMpztNl/view?usp=drive_link)(18GB) from our google drive and save in your local/gdrive.    
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


2. Create a folder named "Raredis" under the data subfolder in the BioGPT path and paste the raw folder [BioGPT/data/raw](https://github.com/shashank140195/Raredis/tree/main/BioGPT/data) folder inside it or alternatively you can choose different raw files from [pre processed](https://github.com/shashank140195/Raredis/tree/main/BioGPT/Preprocessed_data) directory. 
``` 
os.chdir("/content/BioGPT/data")
os.mkdir("Raredis")

# command to copy files created from rel_is_preprocess.py (.pmid, .x and.y files)
%cp -av "content/drive/Mydrive/raw" "/content/BioGPT/data/Raredis/"
```
The file tree should look like this:      

<img width="314" alt="Screenshot 2023-08-24 at 12 41 21 PM" src="https://github.com/shashank140195/Raredis/assets/69673535/98b0f362-9a76-4db4-96e1-6297d9ab6f55">


3. Copy the [Re-Raredis](https://github.com/shashank140195/Raredis/tree/main/BioGPT/RE-Raredis) under the subfolder "examples" in the BioGPT path. This folder contains the bash files to pre-process, train, and infer.
```
%cp -av "content/drive/mydrive/RE-Raredis" "/content/BioGPT/examples/"
```

The file structure should look like this:    

<img width="528" alt="Screenshot 2023-08-24 at 1 00 36 PM" src="https://github.com/shashank140195/Raredis/assets/69673535/7ad81981-ad26-4f5a-a693-9172e05b9804">

4. Run [preprocess.sh](https://github.com/shashank140195/Raredis/blob/main/BioGPT/RE-Raredis/preprocess.sh)  
```
os.chdir("/content/BioGPT/examples/RE-Raredis")
!bash preprocess.sh
```  
The above command will create 1 more folder named "relis-bin" under the same folder as the raw path as shown below:  

<img width="517" alt="Screenshot 2023-08-24 at 1 04 10 PM" src="https://github.com/shashank140195/Raredis/assets/69673535/86aa1ad5-4942-4fb6-add7-2e68eee9095d">

5. Run train.sh to begin training the model. this will create a folder "RE-Raredis-BioGPT" under checkpoint folder. you can change configs in train.sh bash file.
```
!bash train.sh
```  

6. After training run infer.sh. This script runs inference on the test.txt and generates a .detok file
```
!bash infer.sh
```  
<img width="405" alt="Screenshot 2023-08-24 at 1 32 30 PM" src="https://github.com/shashank140195/Raredis/assets/69673535/0b108476-9163-4086-8bb9-e629ba59cd48">  

7. Post-processing  
After inference, run the [BioGPT/scripts/postprocess](https://github.com/shashank140195/Raredis/tree/main/BioGPT/scripts/postprocess) to fetch the inference in the desired JSON format.

8. Evaluation 
Run [BioGPT/scripts/eval/eval_per_rel_type.py](https://github.com/shashank140195/Raredis/tree/main/BioGPT/scripts/eval) to get the overall and individual relation type scores.

## BioMedLM (Former PubMedGPT)
We use [Lambda Labs](https://lambdalabs.com/?matchtype=p&adgroup=55786367910&feeditemid=&loc_interest_ms=&loc_physical_ms=9014313&network=g&device=c&devicemodel=&adposition=&utm_source=google&utm_campaign=Google_Search_Brand&utm_medium=search&utm_term=lambda%20labs&utm_content=308377104950&hsa_acc=1731978716&hsa_cam=1054662654&hsa_grp=55786367910&hsa_ad=308377104950&hsa_src=g&hsa_tgt=kwd-315332575824&hsa_kw=lambda%20labs&hsa_mt=p&hsa_net=adwords&hsa_ver=3&gclid=Cj0KCQjw_5unBhCMARIsACZyzS00NLOnqMDfJtP3WMME-CkkQRYstbA5I_TXGsfx7K2nLb7nMW0bCxQaAnUwEALw_wcB) to train [BioMedLM](https://huggingface.co/stanford-crfm/BioMedLM) by Stanford on 1 A100 GPU along with [Deepspeed](https://github.com/microsoft/DeepSpeed) for CPU offloading.

We follow the same guidelines to prepare data and model training provided at [BioMedLM's author's github](https://github.com/stanford-crfm/BioMedLM/tree/main/finetune) for NLG (Seq2seq) task.  

### 1. Data Prep  
We use the same JSON files we created earlier using [BioGPT/scripts/data_preparation/rawToJSON.py](https://github.com/shashank140195/Raredis/tree/main/BioGPT/scripts/data_preparation) to build the data required for BioMedLM input.  

Run [BioMedLM/scripts/databuilder](https://github.com/shashank140195/Raredis/tree/main/BioMedLM/scripts/databuilder) to build the files required to train BioMedLM. Notice, this python file is similar to [BioGPT/scripts/data_preparation](https://github.com/shashank140195/Raredis/tree/main/BioGPT/scripts/data_preparation) and generates same files but with different extensions. This script will generate split.pmid, split.source, and split.target for train, dev and test repectively as mentioned in the original github repo.

### 2. Configuration & Model Training
Make sure the task dataset is in ./textgen/data. The dataset folder should have <split>.source and <split>.target files. The .source file should contain the original text in a one example per line format and the .target file should contain the desired output in a one example per line format. See example [here](https://github.com/shashank140195/Raredis/tree/main/BioMedLM/data/token_copy_instruction/with_ent_type/rel_is/data/meqsum). Deepspeed config for cpu offloading is present here

Go to ./textgen/gpt2. To finetune, run:
```
deepspeed finetune_for_summarization.py --output_dir /home/ubuntu/BioMedLM/output_dir\
  --model_name_or_path stanford-crfm/BioMedLM \
  --deepspeed /home/ubuntu/BioMedLM/finetune/deepspeed/cpu_offload.json \
  --tokenizer_name stanford-crfm/pubmed_gpt_tokenizer \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --save_strategy steps \
  --do_eval \
  --train_data_file /home/ubuntu/BioMedLM/finetune/textgen/data/meqsum/train.source \
  --eval_data_file /home/ubuntu/BioMedLM/finetune/textgen/data/meqsum/valid.source \
  --max_source_length 510 \
  --train_max_target_length 500 \
  --save_total_limit 5 \
  --overwrite_output_dir \
  --gradient_accumulation_steps 16 \
  --learning_rate 1e-5 \
  --warmup_ratio 0.2 \
  --weight_decay 0.01 \
  --seed 7 \
  --evaluation_strategy steps \
  --eval_steps 50 \
  --num_train_epochs 15 \
  --logging_steps 50 \
  --save_steps 50 \
  --logging_first_step \
  --load_best_model_at_end True \
  --metric_for_best_model eval_loss \
  --greater_is_better True \
  --adam_beta2 0.98
```
After finetuning, run generation on test set by:
```
python -u run_generation_batch.py --max_source_length -1 --length 510 --model_name_or_path={finetune_checkpoint} --num_return_sequences 1 --stop_token [SEP] --tokenizer_name={finetune_checkpoint} --task_mode=meqsum --control_mode=no --tuning_mode finetune --gen_dir /home/ubuntu/BioMedLM/temp500 --batch_size 1 --temperature 1.0
``` 
