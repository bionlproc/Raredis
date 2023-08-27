import json
import pandas as pd
from pandasql import sqldf

train_abstracts = "/Users/Downloads/Chemprot/ChemProt_Corpus/chemprot_training" \
                  "/chemprot_training_abstracts.tsv"
dev_abstracts = "/Users/Downloads/Chemprot/ChemProt_Corpus/chemprot_development" \
                "/chemprot_development_abstracts.tsv"
test_abstracts = "/Users/Downloads/Chemprot/ChemProt_Corpus/chemprot_test_gs/chemprot_test_abstracts_gs" \
                 ".tsv"

train_entities = "/Users/Downloads/Chemprot/ChemProt_Corpus/chemprot_training/chemprot_training_entities.tsv"
dev_entities = "/Users/Downloads/Chemprot/ChemProt_Corpus/chemprot_development/chemprot_development_entities.tsv"
test_entities = "/Users/Downloads/Chemprot/ChemProt_Corpus/chemprot_test_gs/chemprot_test_entities_gs.tsv"

train_relations = "/Users/Downloads/Chemprot/ChemProt_Corpus/chemprot_training/chemprot_training_relations.tsv"
dev_relations = "/Users/Downloads/Chemprot/ChemProt_Corpus/chemprot_development/chemprot_development_relations.tsv"
test_relations = "/Users/Downloads/Chemprot/ChemProt_Corpus/chemprot_test_gs/chemprot_test_relations_gs.tsv"

xug_train_json = "/Users/shashankgupta/Downloads/Chemprot/files/train.json"
xug_dev_json = "/Users/shashankgupta/Downloads/Chemprot/files/dev.json"
xug_test_json = "/Users/shashankgupta/Downloads/Chemprot/files/test.json"


def read(json_file, pred_file=None):
    gold_docs = [json.loads(line) for line in open(json_file)]
    if pred_file is None:
        return gold_docs


xug_train_file = read(json_file=xug_train_json)
xug_dev_file = read(json_file=xug_dev_json)
xug_test_file = read(json_file=xug_test_json)

training_doc_keys = []
dev_doc_keys = []

for x in xug_train_file:
    training_doc_keys.append(x["doc_key"])

for y in xug_dev_file:
    dev_doc_keys.append(y["doc_key"])

column_names = ['doc_key', 'title', 'abstract']
train_abs = pd.read_csv(train_abstracts, sep="\t", header=None)
train_abs.columns = column_names
dev_abs = pd.read_csv(dev_abstracts, sep="\t", header=None)
dev_abs.columns = column_names
combined_abs = train_abs.append(dev_abs)
combined_abs = combined_abs.reset_index(drop=True)
train_abs.drop(train_abs.index, inplace=True)
dev_abs.drop(dev_abs.index, inplace=True)

test_abs = pd.read_csv(test_abstracts, sep="\t", header=None)
test_abs.columns = column_names

column_names = ['doc_key', 'ent_id', 'ent_type', 'start', 'end', 'span']
train_ents = pd.read_csv(train_entities, sep="\t", header=None)
train_ents.columns = column_names
dev_ents = pd.read_csv(dev_entities, sep="\t", header=None)
dev_ents.columns = column_names
combined_ents = train_ents.append(dev_ents)
combined_ents = combined_ents.reset_index(drop=True)
train_ents.drop(train_ents.index, inplace=True)
dev_ents.drop(dev_ents.index, inplace=True)

test_ents = pd.read_csv(test_entities, sep="\t", header=None)
test_ents.columns = column_names

column_names = ['doc_key', 'rel_type', 'misc', 'fine_grained_rel_type', 'arg1', 'arg2']
train_rels = pd.read_csv(train_relations, sep="\t", header=None)
train_rels.columns = column_names
dev_rels = pd.read_csv(dev_relations, sep="\t", header=None)
dev_rels.columns = column_names
combined_rels = train_rels.append(dev_rels)
combined_rels = combined_rels.reset_index(drop=True)
train_rels.drop(train_rels.index, inplace=True)
dev_rels.drop(dev_rels.index, inplace=True)

test_rels = pd.read_csv(test_relations, sep="\t", header=None)
test_rels.columns = column_names

mismatch = 0
unicode = ['\xa0', '\u2002', '\u2005', '\u2009', '\u200a']

for z in training_doc_keys:
    selected_abs = combined_abs[combined_abs['doc_key'] == int(z)]
    selected_ents = combined_ents[combined_ents['doc_key'] == int(z)]
    selected_rels = combined_rels[combined_rels['doc_key'] == int(z)]

    train_abs = train_abs.append(selected_abs)
    train_ents = train_ents.append(selected_ents)
    train_rels = train_rels.append(selected_rels)

train_abs = train_abs.reset_index(drop=True)
train_ents = train_ents.reset_index(drop=True)
train_rels = train_rels.reset_index(drop=True)

for y in dev_doc_keys:
    selected_abs_dev = combined_abs[combined_abs['doc_key'] == int(y)]
    selected_ents_dev = combined_ents[combined_ents['doc_key'] == int(y)]
    selected_rels_dev = combined_rels[combined_rels['doc_key'] == int(y)]

    dev_abs = dev_abs.append(selected_abs_dev)
    dev_ents = dev_ents.append(selected_ents_dev)
    dev_rels = dev_rels.append(selected_rels_dev)

dev_abs = dev_abs.reset_index(drop=True)
dev_ents = dev_ents.reset_index(drop=True)
dev_rels = dev_rels.reset_index(drop=True)


rels_to_keep = ['CPR:3', 'CPR:4', 'CPR:5', 'CPR:6', 'CPR:9']


# change these to create data
# test
table_abs = test_abs
table_ent = test_ents
table_res = test_rels

# # train
# table_abs = train_abs
# table_ent = train_ents
# table_res = train_rels

# dev
# table_abs = dev_abs
# table_ent = dev_ents
# table_res = dev_rels

final_doc = dict()
with open('/Users/shashankgupta/Downloads/Chemprot/BioGPT/test.json', 'w') as outfile:
    for x in range(len(table_abs)):
        doc = table_abs["doc_key"][x]
        print(doc)
        title = table_abs["title"][x]
        abstract = table_abs["abstract"][x]
        context = title + " " + abstract
        context = context.lower()
        for item in unicode:
            context = context.replace(item, u' ')

        query = "SELECT * FROM table_ent WHERE doc_key = '" + str(int(doc)) + "'"
        result = sqldf(query)
        result = result.fillna("NA")

        ent_info_dict = dict()
        for i in range(len(result)):
            ent_id = result["ent_id"][i]
            ent_type = result["ent_type"][i].lower()
            if ent_type == "gene-n" or ent_type == "gene-y":
                ent_type = "gene"
            start = result["start"][i]
            end = result["end"][i]
            span = result["span"][i].lower()
            context_span = context[start: end].lower()

            if span != context_span:
                print("mismatch in doc key ", doc, "anno span:", span, "context_span: ", context_span)
                mismatch += 1

            ent_info_dict[ent_id] = {"type": ent_type, "span": span}

        selected_row = table_res[table_res['doc_key'] == doc]
        selected_row = selected_row[selected_row['rel_type'].isin(rels_to_keep)]
        selected_row = selected_row.reset_index(drop=True)

        triples = []
        output_string = ""
        if len(selected_row) == 0:
            triples.append({"rel": "no relation"})

        else:
            for k in range(len(selected_row)):
                temp = dict()
                arg1_id = (selected_row["arg1"][k])[5:]
                arg2_id = (selected_row["arg2"][k])[5:]

                # with ent_Type
                arg1 = ent_info_dict[arg1_id]["type"] + " " + ent_info_dict[arg1_id]["span"]
                arg2 = ent_info_dict[arg2_id]["type"] + " " + ent_info_dict[arg2_id]["span"]

                # w/o ent_type
                # arg1 = ent_info_dict[arg1_id]["span"]
                # arg2 = ent_info_dict[arg2_id]["span"]

                rel_type = (selected_row["rel_type"][k]).lower()

                temp["rel"] = rel_type
                temp["arg1"] = arg1
                temp["arg2"] = arg2
                triples.append(temp)

        final_doc[int(doc)] = {"abstract": "consider the abstract: $ " + context + "$ from the given abstract, all the "
                                                                              "entities and relations among them are: "}
        final_doc[int(doc)]["triples"] = triples
        final_doc[int(doc)]["pmid"] = int(doc)

    json.dump(final_doc, outfile)


