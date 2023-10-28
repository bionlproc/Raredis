import json
import pandas as pd


def get_token_mention_mapping(predicted_ner, sentence):
    mapping_dict = dict()
    for predicted_tokens in predicted_ner:
        type = predicted_tokens[-1].lower().strip()
        mention = " ".join(sentence[predicted_tokens[0]:predicted_tokens[1] + 1]).strip().lower()
        mapping_dict[tuple([predicted_tokens[0], predicted_tokens[1]])] = [mention, type]

    return mapping_dict


def get_mention_from_mapping(predicted_tokens, mapping):
    predicted_tokens = tuple(predicted_tokens)

    # without entity type
    # mention = mapping[predicted_tokens][0].lower().strip()

    # with entity type
    mention = mapping[predicted_tokens][1].lower().strip() + " " + mapping[predicted_tokens][0].lower().strip()

    return mention


def get_gold_entities_mapping(tokens, entities):
    new_dict = dict()
    types = ["DISEASE", "RAREDISEASE", "SKINRAREDISEASE", "SIGN", "SYMPTOM", "ANAPHOR"]
    for g in entities:
        span_string = ""
        for ent_no, values in g.items():
            for idx in range(len(values)):
                if values[idx] not in types:
                    tokens_span = values[idx]
                    span_string = span_string + " ".join(
                        tokens[tokens_span[0]: tokens_span[1] + 1]).strip().lower() + " "
                else:
                    token_type = values[idx].lower()

        entity_String = token_type + " " + span_string[:-1]
        new_dict[ent_no] = entity_String

    return new_dict


def get_relations(relations, entites_mapping):
    gold_Set = set()
    for rels in relations:
        for rel_type, args in rels.items():
            arg1 = entites_mapping[args[0]]
            arg2 = entites_mapping[args[1]]
            gold_Set.add((arg1, arg2, rel_type.lower()))

    return gold_Set


prediction_file = "/Users/shashankgupta/Documents/Raredis/PURE_Xuguang/PubmedBERT_base/relation/predictions.json"
golden_file = "/Users/shashankgupta/Downloads/test2.json"

gold_files_names = ['Chromosome-6-Partial-Trisomy-6q', 'Mixed-Cryoglobulinemia', 'Tinnitus',
                    'Leukoencephalopathy-with-Brain-Stem-and-Spinal-Cord-Involvement-and-Lactate-Elevation',
                    'Arteriovenous-Malformation', 'Spinocerebellar-Ataxia-with-Axonal-Neuropathy',
                    'Pseudo-Hurler-Polydystrophy', 'Gitelman-Syndrome', 'Tarsal-Tunnel-Syndrome',
                    'C3-Glomerulopathy_-Dense-Deposit-Disease-and-C3-Glomerulonephritis', 'Trichothiodystrophy',
                    'Alpers-Disease', 'Autosomal-Dominant-Hereditary-Ataxia', 'Glutathione-Synthetase-Deficiency',
                    'Epidermolytic-Ichthyosis', 'Banti_s-Syndrome', 'Potter-Syndrome',
                    'SLC13A5-Epileptic-Encephalopathy', 'Meningitis-Tuberculous',
                    'Very-Long-Chain-Acyl-CoA-Dehydrogenase-Deficiency-(LCAD)', 'Enterobiasis', 'Lichen-Sclerosus',
                    'Pediatric-Crohns-Disease', 'Alkaptonuria', 'Primary-Lateral-Sclerosis',
                    'Multifocal-Motor-Neuropathy', 'Alveolar-Soft-Part-Sarcoma',
                    'Anemia-Hereditary-Nonspherocytic-Hemolytic', 'Hartnup-Disease', 'Soft-Tissue-Sarcoma', 'CADASIL',
                    'Berylliosis', 'Cleidocranial-Dysplasia', 'Fetal-Alcohol-Syndrome',
                    'Trisomy-9p-(Multiple-Variants)', 'Glucose', 'Chromosome-9-Ring', 'WHIM-Syndrome',
                    'Heart-Block-Congenital', 'Primary-Central-Nervous-System-Lymphoma', 'Hyperekplexia', 'Dejerine',
                    'Dysplasia-Epiphysealis-Hemimelica', 'Duchenne-Muscular-Dystrophy', 'Ameloblastic-Carcinoma',
                    'Andersen', 'Fetal-Valproate-Syndrome', 'Succinic-Semialdehyde-Dehydrogenase-Deficiency', 'AP',
                    'Chromosome-10-Distal-Trisomy-10q', 'Cystic-Fibrosis', 'Laband-Syndrome', 'Dup15q-Syndrome',
                    'Baller', 'Myopathy-Congenital-Batten-Turner-Type', 'Protein-S-Deficiency',
                    'Binder-Type-Nasomaxillary-Dysplasia', 'Ichthyosis-Lamellar', 'Leukodystrophy',
                    'Creutzfeldt-Jakob-Disease', 'Subacute-Sclerosing-Panencephalitis',
                    'Waldenstrom_s-Macroglobulinemia', 'Hyperferritinemia-Cataract-Syndrome', 'West-Syndrome',
                    'Alagille-Syndrome', 'Elephantiasis', 'Jejunal-Atresia', 'Carnosinemia', 'Mikulicz-Syndrome',
                    'Hypoparathyroidism', 'Pseudoachondroplasia', 'Chromosome-14-Ring', 'Ewing-Sarcoma',
                    'Hypoplastic-Left-Heart-Syndrome', 'Fetal-Hydantoin-Syndrome', 'Arthritis-Infectious',
                    'Townes-Brocks-Syndrome', 'Turner-Syndrome', 'Cornelia-de-Lange-Syndrome', 'Cat-Eye-Syndrome',
                    'Buerger_s-Disease', 'Human-Monocytic-Ehrlichiosis-(HME)', 'Pycnodysostosis',
                    'Empty-Sella-Syndrome', 'Agenesis-of-Corpus-Callosum', 'Barakat-Syndrome', 'Kienbock-Disease',
                    'Alport-Syndrome', 'Acanthosis-Nigricans', 'Schindler-disease', 'Mandibuloacral-Dysplasia',
                    'Classic-Infantile-CLN1-Disease', 'Retroperitoneal-Fibrosis', 'Antiphospholipid-Syndrome',
                    'Floating-Harbor-Syndrome', 'Melorheostosis', 'Monilethrix', 'Fibrous-Dysplasia', 'Meige-Syndrome',
                    'VACTERL-Association', 'Pseudocholinesterase-Deficiency',
                    'Ovotesticular-Disorder-of-Sex-Development', 'Balo-Disease',
                    'Argininie_-Glycine-Amidinotransferase-Deficiency']


def read(json_file, pred_file=None):
    gold_docs = [json.loads(line) for line in open(json_file)]
    if pred_file is None:
        return gold_docs


pred = read(prediction_file, pred_file=None)

new_list = []
for x in pred:
    new_pred = dict()
    new_pred["doc_key"] = x["doc_key"][:-6]
    new_pred["sentences"] = x["sentences"]
    new_pred["ner"] = x["ner"]
    new_pred["relations"] = x["relations"]
    new_pred["predicted_ner"] = x["predicted_ner"]
    new_pred["predicted_relations"] = x["predicted_relations"]
    new_list.append(new_pred)

gold = read(golden_file, pred_file=None)

gold_dict = dict()
for z in gold:
    tokens = z["tokens"]
    entities = z["entities"]
    if "relations" in z:
        relations = z["relations"]
        entites_mapping = get_gold_entities_mapping(tokens, entities)
        gold_relations = get_relations(relations, entites_mapping)
        gold_dict[z["doc"]] = gold_relations

    else:
        print("no relation")
        gold_dict[z["doc"]] = set()

tp, fp, fn = 0, 0, 0

produce_fp, produce_tp, produce_fn = 0, 0, 0
is_a_fp, is_a_tp, is_a_fn = 0, 0, 0
iro_fp, iro_tp, iro_fn = 0, 0, 0
syn_fp, syn_tp, syn_fn = 0, 0, 0
acro_fp, acro_tp, acro_fn = 0, 0, 0
anaphora_fp, anaphora_tp, anaphora_fn = 0, 0, 0

columns = ['doc_name', "text", 'fp_prediction', "fn_prediction"]
df = pd.DataFrame(columns=columns)

for k in new_list:
    prediction_Set = set()
    prediction_Set_is_a = set()
    prediction_Set_produce = set()
    prediction_Set_anaphora = set()
    prediction_Set_iro = set()
    prediction_Set_synon = set()
    prediction_Set_acron = set()

    doc_key = k["doc_key"]
    sentence_tokens = k["sentences"][0]
    predicted_ner = k["predicted_ner"][0]
    mapping = get_token_mention_mapping(predicted_ner, sentence_tokens)
    predicted_rels = k["predicted_relations"][0]

    for predictions in predicted_rels:
        pred_arg1_tokens = predictions[:2]
        pred_arg1 = get_mention_from_mapping(pred_arg1_tokens, mapping)

        predicted_rel_type = predictions[-1].strip().lower()

        pred_arg2_tokens = predictions[2:4]
        pred_arg2 = get_mention_from_mapping(pred_arg2_tokens, mapping)

        prediction_Set.add((pred_arg1, pred_arg2, predicted_rel_type))

        if predicted_rel_type == "is_a":
            prediction_Set_is_a.add((pred_arg1, pred_arg2, predicted_rel_type))

        elif predicted_rel_type == "produces":
            prediction_Set_produce.add((pred_arg1, pred_arg2, predicted_rel_type))

        elif predicted_rel_type == "anaphora":
            prediction_Set_anaphora.add((pred_arg1, pred_arg2, predicted_rel_type))

        elif predicted_rel_type == "increases_risk_of":
            prediction_Set_iro.add((pred_arg1, pred_arg2, predicted_rel_type))

        elif predicted_rel_type == "is_synon":
            prediction_Set_synon.add((pred_arg1, pred_arg2, predicted_rel_type))

        elif predicted_rel_type == "is_acron":
            prediction_Set_acron.add((pred_arg1, pred_arg2, predicted_rel_type))

    gold_set = gold_dict[doc_key]
    gold_Set_is_a = set()
    gold_Set_produce = set()
    gold_Set_anaphora = set()
    gold_Set_iro = set()
    gold_Set_synon = set()
    gold_Set_acron = set()

    for gr in gold_set:
        rels_type = gr[2]

        if rels_type == "is_a":
            gold_Set_is_a.add(gr)

        elif rels_type == "produces":
            gold_Set_produce.add(gr)

        elif rels_type == "anaphora":
            gold_Set_anaphora.add(gr)

        elif rels_type == "increases_risk_of":
            gold_Set_iro.add(gr)

        elif rels_type == "is_synon":
            gold_Set_synon.add(gr)

        elif rels_type == "is_acron":
            gold_Set_acron.add(gr)

    # print("fp: ", prediction_Set - gold_set)
    # print("fn: ", gold_set - prediction_Set)
    # print("tp: ", prediction_Set.intersection(gold_set))
    # print("\n")

    fp += len(prediction_Set - gold_set)
    fn += len(gold_set - prediction_Set)
    tp += len(prediction_Set.intersection(gold_set))

    produce_fp += len(prediction_Set_produce - gold_Set_produce)
    produce_tp += len(prediction_Set_produce.intersection(gold_Set_produce))
    produce_fn += len(gold_Set_produce - prediction_Set_produce)

    is_a_fp += len(prediction_Set_is_a - gold_Set_is_a)
    is_a_tp += len(prediction_Set_is_a.intersection(gold_Set_is_a))
    is_a_fn += len(gold_Set_is_a - prediction_Set_is_a)

    iro_fp += len(prediction_Set_iro - gold_Set_iro)
    iro_tp += len(prediction_Set_iro.intersection(gold_Set_iro))
    iro_fn += len(gold_Set_iro - prediction_Set_iro)

    syn_fp += len(prediction_Set_synon - gold_Set_synon)
    syn_tp += len(prediction_Set_synon.intersection(gold_Set_synon))
    syn_fn += len(gold_Set_synon - prediction_Set_synon)

    acro_fp += len(prediction_Set_acron - gold_Set_acron)
    acro_tp += len(prediction_Set_acron.intersection(gold_Set_acron))
    acro_fn += len(gold_Set_acron - prediction_Set_acron)

    anaphora_fp += len(prediction_Set_anaphora - gold_Set_anaphora)
    anaphora_tp += len(prediction_Set_anaphora.intersection(gold_Set_anaphora))
    anaphora_fn += len(gold_Set_anaphora - prediction_Set_anaphora)

    text = ""
    for abc in gold:
        if abc["doc"] == doc_key:
            text = abc["text"]
    columns = ['doc_name', "text", 'fp_prediction', "fn_prediction"]
    new_rows = [{'doc_name': doc_key, "text": text, 'fp_prediction': prediction_Set - gold_set,
                 'fn_prediction': gold_set - prediction_Set}]
    df = df.append(new_rows, ignore_index=True)

df.to_csv("/Users/shashankgupta/Documents/Raredis/PURE_Xuguang/PubmedBERT_base/Pipeline_error_analysis_2.csv",
          index=False)

p = tp / (tp + fp)
r = tp / (tp + fn)
print("Precision: ", p)
print("Recall: ", r)
print("fscore: ", 2 * p * r / (p + r))

# Produce Scores
P_prod = produce_tp / (produce_tp + produce_fp)
R_prod = produce_tp / (produce_tp + produce_fn)
Fscore_prod = 2 * P_prod * R_prod / (P_prod + R_prod)

print("Produce precision is: ", P_prod)
print("Produce Recall is: ", R_prod)
print("Produce F-score is: ", Fscore_prod)

# Anaphora Scores
P_ana = anaphora_tp / (anaphora_tp + anaphora_fp)
R_ana = anaphora_tp / (anaphora_tp + anaphora_fn)
Fscore_ana = 2 * P_ana * R_ana / (P_ana + R_ana)

print("Anaphora precision is: ", P_ana)
print("Anaphora Recall is: ", R_ana)
print("Anaphora F-score is: ", Fscore_ana)

# is_a Scores
P_is_a = is_a_tp / (is_a_tp + is_a_fp)
R_is_a = is_a_tp / (is_a_tp + is_a_fn)
Fscore_is_a = 2 * P_is_a * R_is_a / (P_is_a + R_is_a)

print("is_a precision is: ", P_is_a)
print("is_a Recall is: ", R_is_a)
print("is_a F-score is: ", Fscore_is_a)

# is_acron Scores
P_is_acron = acro_tp / (acro_tp + acro_fp)
R_is_acron = acro_tp / (acro_tp + acro_fn)
Fscore_is_acron = 2 * P_is_acron * R_is_acron / (P_is_acron + R_is_acron)

print("is_acron precision is: ", P_is_acron)
print("is_acron Recall is: ", R_is_acron)
print("is_acron F-score is: ", Fscore_is_acron)

# is_synon Scores
P_is_synon = syn_tp / (syn_tp + syn_fp)
R_is_synon = syn_tp / (syn_tp + syn_fn)
Fscore_is_synon = 2 * P_is_synon * R_is_synon / (P_is_synon + R_is_synon)

print("is_synon precision is: ", P_is_synon)
print("is_synon Recall is: ", R_is_synon)
print("is_synon F-score is: ", Fscore_is_synon)

# increase_risk_of Scores
P_iro = iro_tp / (iro_tp + iro_fp)
R_iro = iro_tp / (iro_tp + iro_fn)
Fscore_iro = 2 * P_iro * R_iro / (P_iro + R_iro)

print("increase_risk_of precision is: ", P_iro)
print("increase_risk_of Recall is: ", R_iro)
print("increase_risk_of F-score is: ", Fscore_iro)

# overall check
total_tp = produce_tp + is_a_tp + iro_tp + syn_tp + acro_tp + anaphora_tp
total_fp = produce_fp + is_a_fp + iro_fp + syn_fp + acro_fp + anaphora_fp
total_fn = produce_fn + is_a_fn + iro_fn + syn_fn + acro_fn + anaphora_fn

overall_P = total_tp / (total_tp + total_fp)
overall_R = total_tp / (total_tp + total_fn)
overall_F = 2 * overall_P * overall_R / (overall_P + overall_R)

print(overall_P, overall_R, overall_F)
