from seq2rel.common import util
from seq2rel.seq2rel import Seq2Rel
import pandas as pd

columns = ['Text', "gold_rels", "pred_rels", "rel_type", 'FP', 'FN']
df = pd.DataFrame(columns=columns)

# Add path to your fine-tuned model
model = 'path_to_your_model/model.tar.gz'
seq2rel = Seq2Rel(model)
from allennlp.common.file_utils import cached_path

true_positive_sum, pred_sum, true_sum = 0, 0, 0
false_negative_sum, false_positive_sum = 0,0
predictions=[]
gold=[]

produce_fp, produce_tp, produce_fn = 0, 0, 0
is_a_fp, is_a_tp, is_a_fn = 0, 0, 0
iro_fp, iro_tp, iro_fn = 0, 0, 0
syn_fp, syn_tp, syn_fn = 0, 0, 0
acro_fp, acro_tp, acro_fn = 0, 0, 0
anaphora_fp, anaphora_tp, anaphora_fn = 0, 0, 0

with open(cached_path('home/seq2rel/data/test.txt'), "r") as data_file:  # add path to gold test file
    for line_num, line in enumerate(data_file):
        line = line.strip("\n")
        line_parts = line.split('\t')
        input_text = line_parts[0]
        gold.append(line_parts[1])
        gold_relations = [line_parts[1]]
        predicted_relations = seq2rel(input_text)
        predicted_relations = [i.replace(' - ', '-') for i in predicted_relations]
        predicted_relations = [i.replace('( ', '(') for i in predicted_relations]
        predicted_relations = [i.replace(' )', ')') for i in predicted_relations]
        predicted_relations = [i.replace(' / ', '/') for i in predicted_relations]

        predictions.append(predicted_relations)
        gold_annotations = util.extract_relations(gold_relations, remove_duplicate_ents=True)
        pred_annotations = util.extract_relations(predicted_relations, remove_duplicate_ents=True)

        for pred_ann, gold_ann in zip(pred_annotations, gold_annotations):
            gold_rel_types = list(gold_ann.keys())
            pred_rel_types = list(pred_ann.keys())

            for i in range(len(gold_rel_types)):
                if i ==0 :
                    new_row = {'Text': input_text, "gold_rels": gold_rel_types, "pred_rels": pred_rel_types, "rel_type": "", 'FP': "", 'FN': ""}
                    df = df.append(new_row, ignore_index=True)
                key = gold_rel_types[i]
                pred_rels = pred_ann.get(key, [])
                pred_sum += len(pred_rels)
                gold_rels = gold_ann.get(key, [])

                common_list = [c for c in gold_rels if c in pred_rels]
                false_negatives = [c for c in gold_rels if c not in pred_rels]
                false_positives = [c for c in pred_rels if c not in gold_rels]

                false_negative_sum += len(false_negatives)
                false_positive_sum += len(false_positives)

                # Add a new row
                new_row = {'Text': "", "gold_rels": "", "pred_rels": "", "rel_type": key, 'FP': false_positives, 'FN': false_negatives}
                df = df.append(new_row, ignore_index=True)
                true_positive_sum += len(common_list)
                true_sum += len(gold_rels)

                if key == "Produces":
                    produce_fp += len(false_positives)
                    produce_tp += len(common_list)
                    produce_fn += len(false_negatives)

                elif key == "Anaphora":
                    anaphora_fp += len(false_positives)
                    anaphora_tp += len(common_list)
                    anaphora_fn += len(false_negatives)

                elif key == "Is_a":
                    is_a_fp += len(false_positives)
                    is_a_tp += len(common_list)
                    is_a_fn += len(false_negatives)

                elif key == "Is_acron":
                    acro_fp += len(false_positives)
                    acro_tp += len(common_list)
                    acro_fn += len(false_negatives)

                elif key == "Is_synon":
                    syn_fp += len(false_positives)
                    syn_tp += len(common_list)
                    syn_fn += len(false_negatives)

                elif key == "Increases_risk_of":
                    iro_fp += len(false_positives)
                    iro_tp += len(common_list)
                    iro_fn += len(false_negatives)

            rels_not_in_gold = [k for k in pred_rel_types if k not in gold_rel_types]
            for k in range(len(rels_not_in_gold)):
                fp_rel = rels_not_in_gold[k]
                new_row = {'Text': "", "gold_rels": "", "pred_rels": "", "rel_type": fp_rel, 'FP': pred_ann.get(fp_rel, []), 'FN': ""}
                df = df.append(new_row, ignore_index=True)
                pred_sum += len(pred_ann.get(fp_rel, []))
                false_positive_sum += len(pred_ann.get(fp_rel, []))

                if fp_rel == "Produces":
                    produce_fp += len(pred_ann.get(fp_rel, []))

                elif fp_rel == "Anaphora":
                    anaphora_fp += len(pred_ann.get(fp_rel, []))

                elif fp_rel == "Is_a":
                    is_a_fp += len(pred_ann.get(fp_rel, []))

                elif fp_rel == "Is_acron":
                    acro_fp += len(pred_ann.get(fp_rel, []))

                elif fp_rel == "Is_synon":
                    syn_fp += len(pred_ann.get(fp_rel, []))

                elif fp_rel == "Increases_risk_of":
                    iro_fp += len(pred_ann.get(fp_rel, []))

# Dump DataFrame to CSV file
df.to_csv('home/detailed_prediction.csv', index=False)


# Produce Scores
P_prod = produce_tp/(produce_tp + produce_fp)
R_prod = produce_tp/(produce_tp + produce_fn)
Fscore_prod = 2*P_prod*R_prod/(P_prod+R_prod)

print("Produce precision is: ", P_prod)
print("Produce Recall is: ", R_prod)
print("Produce F-score is: ", Fscore_prod)

# Anaphora Scores
P_ana = anaphora_tp/(anaphora_tp + anaphora_fp)
R_ana = anaphora_tp/(anaphora_tp + anaphora_fn)
Fscore_ana = 2*P_ana*R_ana/(P_ana+R_ana)

print("Anaphora precision is: ", P_ana)
print("Anaphora Recall is: ", R_ana)
print("Anaphora F-score is: ", Fscore_ana)

# is_a Scores
P_is_a = is_a_tp/(is_a_tp + is_a_fp)
R_is_a = is_a_tp/(is_a_tp + is_a_fn)
Fscore_is_a = 2*P_is_a*R_is_a/(P_is_a+R_is_a)

print("is_a precision is: ", P_is_a)
print("is_a Recall is: ", R_is_a)
print("is_a F-score is: ", Fscore_is_a)

# is_acron Scores
P_is_acron = acro_tp/(acro_tp + acro_fp)
R_is_acron = acro_tp/(acro_tp + acro_fn)
Fscore_is_acron = 2*P_is_acron*R_is_acron/(P_is_acron+R_is_acron)

print("is_acron precision is: ", P_is_acron)
print("is_acron Recall is: ", R_is_acron)
print("is_acron F-score is: ", Fscore_is_acron)

# is_synon Scores
P_is_synon = syn_tp/(syn_tp + syn_fp)
R_is_synon = syn_tp/(syn_tp + syn_fn)
Fscore_is_synon = 2*P_is_synon*R_is_synon/(P_is_synon+R_is_synon)

print("is_synon precision is: ", P_is_synon)
print("is_synon Recall is: ", R_is_synon)
print("is_synon F-score is: ", Fscore_is_synon)

# increase_risk_of Scores
P_iro = iro_tp/(iro_tp + iro_fp)
R_iro = iro_tp/(iro_tp + iro_fn)
Fscore_iro = 2*P_iro*R_iro/(P_iro+R_iro)

print("increase_risk_of precision is: ", P_iro)
print("increase_risk_of Recall is: ", R_iro)
print("increase_risk_of F-score is: ", Fscore_iro)

# Overall Score
P = true_positive_sum/pred_sum
R = true_positive_sum/true_sum
Fscore = 2*P*R/(P+R)

print("Overall precision is: ", P)
print("Overall Recall is: ", R)
print("Overall F-score is: ", Fscore)

# Overall Score
P_form = true_positive_sum/ (true_positive_sum + false_positive_sum)
R_form = true_positive_sum/ (true_positive_sum + false_negative_sum)
Fscore_form = 2*P_form*R_form/(P_form+R_form)

print("Overall (Formula) precision is: ", P_form)
print("Overall (Formula) Recall is: ", R_form)
print("Overall (Formula) F-score is: ", Fscore_form)