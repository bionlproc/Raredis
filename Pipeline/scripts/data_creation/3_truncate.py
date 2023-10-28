import os, string
import warnings
from os.path import basename, splitext
from transformers import AutoTokenizer
from typing import List
import re
import pandas as pd
import spacy

use_scispacy = True
nlp_name = "en_core_sci_lg" if use_scispacy else "en_core_web_sm"
nlp = spacy.load(nlp_name)
max_token_cap = 500


def extract_indices_from_brat_annotation(indices: str) -> List[int]:
    indices = re.findall(r"\d+", indices)
    indices = sorted([int(i) for i in indices])
    return indices

text_path = "/Users/shashankgupta/Desktop/BioGPT_inputTest/Raredis_corrected_annotation/final_corrections/test/input_text"
ann_path = "/Users/shashankgupta/Desktop/BioGPT_inputTest/Raredis_corrected_annotation/final_corrections/test/ann"

count_large_doc = 0
columns = ['doc_name', "original_doc_size", "modified_doc_size", "tot_ent_org_doc", 'entities_dropped', "tot_rels_org_doc", 'relations_dropped']
df = pd.DataFrame(columns=columns)

# CHECKING TEXT LENGTH TO FIT TO BERT MAX TOKEN SIZE
vocab_name = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
tokenizer = AutoTokenizer.from_pretrained(vocab_name)

for document in os.listdir(ann_path):
    EntitiesDroppedRangeLimit = 0
    relationsDroppedRangeLimit = 0
    largeDocFlag = 0
    entity_count = 0
    total_relations = 0
    maxTokensReachedEnt = []
    info_dict = {}

    with open(os.path.join(ann_path, document), "r") as in_f:
        doc_name = splitext(basename(document))[1]
        if doc_name == ".ann":
            document = document.replace(".ann", "")
            # if document != "Fibromuscular-Dysplasia":
            #     continue
            with open(os.path.join("/Users/shashankgupta/Downloads/PURE_test/test/ann", "%s.ann" % document), 'w') as fa:
                text = open(os.path.join(text_path, "%s.txt" % document)).read()
                doc = nlp(text)
                bert_tokens = []
                # Check the number of tokens in the sentence
                for token in doc:
                    Ttext = token.text
                    sub_tokens = tokenizer.tokenize(Ttext)
                    bert_tokens += sub_tokens

                # bert_tokens = tokenizer.tokenize(text)
                if len(bert_tokens) > max_token_cap:
                    info_dict['doc_name'] = document
                    info_dict["original_doc_size"] = len(bert_tokens)
                    count_large_doc += 1
                    largeDocFlag = 1
                    truncated_doc = bert_tokens[:500]
                    reference_doc = tokenizer.convert_tokens_to_string(truncated_doc)

                    punctuation_count = 0
                    for char in reference_doc:
                        if char in string.punctuation:
                            punctuation_count += 1

                    length = len(reference_doc) - punctuation_count
                    text = text[:length]
                    info_dict["modified_doc_size"] = len(tokenizer.tokenize(text))

                for line in in_f:
                    if line[0] == "T":
                        entity_count +=1
                        sp = line.split("\t")
                        sp1 = sp[1].split(" ")
                        indices = sorted(extract_indices_from_brat_annotation(" ".join(sp1[1:])), reverse= True)
                        if largeDocFlag == 1:
                            if int(indices[0]) >= length:
                                maxTokensReachedEnt.append(int(sp[0][1:]))
                                warnings.warn(
                                    f'(Entity: {sp[0]}) falls outside the BERT input range of document {document} and needs to be dropped "{sp[2]}.')
                                EntitiesDroppedRangeLimit += 1
                            else:
                                fa.write("\t".join(sp))
                        else:
                            fa.write("\t".join(sp))

                    elif line[0] == 'R':
                        total_relations = total_relations + 1
                        spr = line.split()
                        spr1 = line.split("\t")
                        spr2 = spr1[1].split()
                        entity1 = spr2[1]
                        entity1_id = int(entity1[6:])

                        entity2 = spr2[2]
                        entity2_id = int(entity2[6:])

                        if entity1_id in maxTokensReachedEnt or entity2_id in maxTokensReachedEnt:
                            warnings.warn(f'Relation {spr[0]} needs to be dropped. '
                                          f'It contains entities which falls out of the BERT range.')
                            relationsDroppedRangeLimit += 1

                        else:
                            fa.write(line)

    if largeDocFlag == 1:
        info_dict["tot_ent_org_doc"] = entity_count
        info_dict["entities_dropped"] = EntitiesDroppedRangeLimit
        info_dict["tot_rels_org_doc"] = total_relations
        info_dict["relations_dropped"] = relationsDroppedRangeLimit
        new_rows = [info_dict]
        df = df.append(new_rows, ignore_index=True)

    with open(os.path.join("/Users/shashankgupta/Downloads/PURE_test/test/input_text", "%s.txt" % document), 'w') as f:
        f.write(text)

df.to_csv("/Users/shashankgupta/Downloads/PURE_test/test/test_truncated_doc_stats.csv", index=False)

