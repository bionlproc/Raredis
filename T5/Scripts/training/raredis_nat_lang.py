# import pydevd_pycharm
# #
# pydevd_pycharm.settrace('localhost', port=12345, stdoutToServer=True, stderrToServer=True)

import numpy as np
import sys
import re
import pandas as pd
import json
import torch
import gc

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, \
    HfArgumentParser, set_seed, DataCollatorForSeq2Seq, EarlyStoppingCallback
from typing import Optional
from dataclasses import dataclass, field
from datasets import Dataset


sys.path.insert(0, "../../../../../..")

global_tokenizer = None


def get_original_ent_type(enttype):
    enttype = enttype.replace("rare disease", "raredisease")
    enttype = enttype.replace("rare skin disease", "skinraredisease")
    enttype = enttype.replace("signs", "sign")

    return enttype


def convert_natlang_sentence(sentence):
    sentence = get_original_ent_type(sentence)
    ans = None
    produces = re.match(r"^(.*) is a (.*?) that produces (.*), as a (.*)", sentence)
    synonym = re.match(r"the (.*?) (.*) and the (.*?) (.*) are synonyms", sentence)
    anaphora = re.match(r"the term (.*?) is an anaphor that refers back to the entity of the (.*?) (.*)", sentence)
    acronym = re.match(r"the acronym (.*) stands for (.*), a (.*)", sentence)
    increases_risk_of = re.match(r"the presence of the (.*?) (.*) increases the risk of developing the (.*?) (.*)",
                                 sentence)
    is_a = re.match(r"the (.*?) (.*) is a type of (.*), a (.*)", sentence)
    if produces is not None:
        produces = produces.groups()
        enttype1 = produces[1].strip()
        enttype2 = produces[3].strip()
        ans = (enttype1 + " " + produces[0].strip(), "produces", enttype2 + " " + produces[2].strip())
    elif synonym is not None:
        synonym = synonym.groups()
        enttype1 = synonym[0].strip()
        enttype2 = synonym[2].strip()
        ans = (enttype1 + " " + synonym[1].strip(), "is_synon", enttype2 + " " + synonym[3].strip())
    elif anaphora is not None:
        anaphora = anaphora.groups()
        enttype1 = "anaphor"
        enttype2 = anaphora[1].strip()
        ans = (enttype2 + " " + anaphora[2].strip(), "anaphora", enttype1 + " " + anaphora[0].strip("'"))
    elif acronym is not None:
        acronym = acronym.groups()
        enttype2 = acronym[2].strip()
        enttype1 = enttype2
        ans = (enttype1 + " " + acronym[0].strip(), "is_acron", enttype2 + " " + acronym[1].strip())
    elif increases_risk_of is not None:
        increases_risk_of = increases_risk_of.groups()
        enttype1 = increases_risk_of[0].strip()
        enttype2 = increases_risk_of[2].strip()
        ans = (enttype1 + " " + increases_risk_of[1].strip(), "increases_risk_of",
               enttype2 + " " + increases_risk_of[3].strip())
    elif is_a is not None:
        is_a = is_a.groups()
        enttype1 = is_a[0].strip()
        enttype2 = is_a[3].strip()
        ans = (enttype1 + " " + is_a[1].strip(), "is_a", enttype2 + " " + is_a[2].strip())

    elif sentence.strip() == "there are no relations in the abstract":
        ans = ("", "no relations", "")

    return ans


def split_sentence(line):
    sentences = re.split(r"; ", line)
    return list(set(sentences))


def do_eval(preds, golden):

    num_missing = 0
    true_positive_sum = 0
    fp = 0
    fn = 0
    tn = 0

    columns = ['doc_name', 'gold_rels', 'pred_rels', 'rel_type', 'fp_prediction', "fn_prediction"]
    df = pd.DataFrame(columns=columns)

    for gold,pred in zip(golden, preds):
        gold_arg1_set, gold_arg2_set, gold_rel_set, gold_set = set(), set(), set(), set()
        pred_arg1_set, pred_arg2_set, pred_rel_set, pred_set = set(), set(), set(), set()
        gold_rel = ""
        for tp in gold:
            gold_rel = tp[1].strip().lower()
            if gold_rel != "no relations":
                arg1 = tp[0].strip().lower()
                arg2 = tp[2].strip().lower()
                gold_arg1_set.add(arg1)
                gold_arg2_set.add(arg2)
                gold_rel_set.add(gold_rel)
                gold_set.add((arg1, arg2, gold_rel))

        if pred:
            for p_tp in pred:
                rel = p_tp[1].strip().lower()
                if rel == "no relations" and gold_rel == "no relations":
                    tn += 1
                    continue

                elif gold_rel == "no relations" and rel != "no relations":
                    fp += len(pred_set)
                    continue

                elif gold_rel != "no relations" and rel == "no relations":
                    fn += len(gold_set)
                    continue

                arg1 = p_tp[0].strip().lower()
                arg2 = p_tp[2].strip().lower()
                pred_arg1_set.add(arg1)
                pred_arg2_set.add(arg2)
                pred_rel_set.add(rel)
                pred_set.add((arg1, arg2, rel))

        fp_set = pred_set - gold_set
        fn_set = gold_set - pred_set

        # print("Prediction set: ", pred_set)
        # print("gold set: ", gold_set)
        # print("False positive set: ", fp_set)
        # print("False negative set: ", fn_set)
        # print("\n")
        fp += len(pred_set - gold_set)
        fn += len(gold_set - pred_set)

        common_triples = gold_set.intersection(pred_set)
        true_positive_sum += len(common_triples)

    R = ((true_positive_sum / (true_positive_sum + fn)) if (true_positive_sum + fn)!=0 else 0)
    P = ((true_positive_sum / (true_positive_sum + fp)) if (true_positive_sum + fp)!=0 else 0)

    Fscore = (((2 * P * R) / (P + R)) if (P+R)!=0 else 0)
    return P, R, Fscore


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    labels = np.where(labels != -100, labels, global_tokenizer.pad_token_id)
    predictions = np.where(predictions != -100, predictions, global_tokenizer.pad_token_id)
    preds = []
    golden = []

    for i, x in enumerate(zip(predictions, labels)):
        ret = []
        gold = []
        all_lines = []
        all_pred_lines = []

        prediction = global_tokenizer.decode(x[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        # print(prediction)
        print("length of prediction: ", len(global_tokenizer.tokenize(prediction)))
        pred_sentences = split_sentence(prediction)

        for linep in pred_sentences:
            e = linep.strip()
            if len(e) > 0 and e[-1] == ".":
                all_pred_lines.append(e[:-1])
            else:
                all_pred_lines.append(e)

        label = global_tokenizer.decode(x[1], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        gold_sentences = split_sentence(label)
        for line in gold_sentences:
            e = line.strip()
            if len(e) > 0 and e[-1] == ".":
                all_lines.append(e[:-1])
            else:
                all_lines.append(e)

        for sen in all_lines:
            gold_ans = convert_natlang_sentence(sen)
            if gold_ans is not None:
                gold.append(gold_ans)
        golden.append(gold)

        for pred_Sen in all_pred_lines:
            pred_ans = convert_natlang_sentence(pred_Sen)
            if pred_ans is not None:
                ret.append(pred_ans)
        preds.append(ret)
    P, R, f1 = do_eval(preds, golden)
    return {"P": P, "R": R, "f1": f1}


@dataclass
class ModelArguments:
    """
    Arguments for the model
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Leave None if you want to train a model from"
                " scratch."
            )
        },
    )

    tokenizer_name: Optional[str] = field(
        default="gpt2", metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )

    use_flash: bool = field(
        default=False, metadata={"help": "Use flash attention."}
    )


@dataclass
class DataArguments:
    """
    Arguments for data
    """

    train_data_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a text file)."}
    )
    eval_data_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    max_source_length: Optional[int] = field(
        default=510, metadata={"help": "the max source length of summarization data. "}
    )
    train_max_target_length: Optional[int] = field(
        default=510, metadata={"help": "the max target length for training data. "}
    )
    eval_max_target_length: Optional[int] = field(
        default=510, metadata={"help": "the max target length for dev data. "}
    )
    seq_prefix: Optional[str] = field(
        default="",
        metadata={"help": "A string to begin every sequence with."},
    )
    no_sep: bool = field(
        default=False, metadata={"help": "Don't use a separator token."}
    )
    block_size: int = field(
        default=-1,
        metadata={
            "help": (
                "Optional input sequence length after tokenization."
                "The training dataset will be truncated in block of this size for training."
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )


# Load training data from JSON file
with open('/home/ubuntu/T5/data/nocopy_nat_lang/train.json', 'r') as file:
    train_data = json.load(file)

# Separate source and target texts
train_source_texts = [item['source'] for item in train_data]
train_target_texts = [item['target'] for item in train_data]

# Load validation data from JSON file
with open('/home/ubuntu/T5/data/nocopy_nat_lang/valid.json', 'r') as file:
    valid_data = json.load(file)

# Separate source and target texts
valid_source_texts = [item['source'] for item in valid_data]
valid_target_texts = [item['target'] for item in valid_data]


# noinspection DuplicatedCode
def finetune():
    # parse args
    parser = HfArgumentParser((ModelArguments, DataArguments, Seq2SeqTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # set seed
    set_seed(42)
    torch.cuda.empty_cache()

    # set up model
    # model_id = "t5-3B"
    # model_id = "google/flan-t5-large"
    # model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    model_id = "google/flan-t5-xl"
    # model = AutoModelForSeq2SeqLM.from_pretrained(model_id, device_map = "auto", torch_dtype=torch.bfloat16)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    global global_tokenizer
    global_tokenizer = tokenizer

    # Tokenize source texts and calculate lengths
    train_source_lengths = max([len(tokenizer.tokenize(text)) for text in train_source_texts])
    print("max train source length: ", train_source_lengths)
    valid_source_lengths = max([len(tokenizer.tokenize(text)) for text in valid_source_texts])
    print("max valid source length: ", valid_source_lengths)

    # Tokenize target texts and calculate lengths
    train_target_lengths = max([len(tokenizer.tokenize(text)) for text in train_target_texts])
    print("max train target length: ", train_target_lengths)
    valid_target_lengths = max([len(tokenizer.tokenize(text)) for text in valid_target_texts])
    print("max valid target length: ", valid_target_lengths)

    max_input_length = max(train_source_lengths, valid_source_lengths)
    max_target_length = max(train_target_lengths, valid_target_lengths)

    # we want to ignore tokenizer pad token in the loss
    label_pad_token_id = -100
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        padding="longest",
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8,
        max_length=2048,
        return_tensors="pt"
    )

    train_tokenized_data = tokenizer(train_source_texts, max_length=1024, padding="max_length", truncation=True, return_tensors="pt")
    train_target_tokenized_data = tokenizer(train_target_texts, max_length=1024, padding="max_length", truncation=True, return_tensors="pt")

    # 3. Convert the tokenized data into the HuggingFace dataset format
    train_dataset_dict = {
        "input_ids": train_tokenized_data["input_ids"].numpy(),
        "attention_mask": train_tokenized_data["attention_mask"].numpy(),
        "labels": train_target_tokenized_data["input_ids"].numpy()  # copying input_ids to labels
    }

    # 4. Convert the dictionary into a HuggingFace Dataset
    train_dataset = Dataset.from_dict(train_dataset_dict)

    valid_tokenized_data = tokenizer(valid_source_texts, max_length=1024, padding="max_length", truncation=True, return_tensors="pt")
    valid_target_tokenized_data = tokenizer(valid_target_texts, max_length=1024, padding="max_length", truncation=True, return_tensors="pt")

    # 3. Convert the tokenized data into the HuggingFace dataset format
    valid_dataset_dict = {
        "input_ids": valid_tokenized_data["input_ids"].numpy(),
        "attention_mask": valid_tokenized_data["attention_mask"].numpy(),
        "labels": valid_target_tokenized_data["input_ids"].numpy()  # copying input_ids to labels
    }

    # 4. Convert the dictionary into a HuggingFace Dataset
    valid_dataset = Dataset.from_dict(valid_dataset_dict)

    # Create the trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
    )

    # Train the model
    trainer.train()
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    finetune()