import re
import json
import pandas as pd

gold_file = "/Users/shashankgupta/Documents/Raredis/BioMedLM/generated_dir_6_550_high/test_gold.txt"
prediction_file = "/Users/shashankgupta/Documents/Raredis/BioMedLM/generated_dir_6_550_high/test_beam.txt"


def split_sentence(line):
    sentences = re.split(r"; ", line)
    return list(set(sentences))


def convert_relis_sentence(sentence):
    ans = None
    segs = re.match(r"the relationship between (.*?) and (.*?) is (.*)", sentence)
    if segs is not None:
        segs = segs.groups()
        relation = segs[2].strip()
        rel = ''
        if relation == "hyponym":
            rel = "is_a"
        elif relation == "producer":
            rel = "produces"
        elif relation == "synonyms":
            rel = "is_synon"
        elif relation == "acronym":
            rel = "is_acron"
        elif relation == "heightens":
            rel = "increases_risk_of"
        elif relation == "antecedent":
            rel = "anaphora"

        ans = (segs[0].strip(), rel, segs[1].strip())

    elif sentence == "no relations":
        ans = ("", "no relations", "")
    return ans


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

        print("Prediction set: ", pred_set)
        print("gold set: ", gold_set)
        print("False positive set: ", fp_set)
        print("False negative set: ", fn_set)
        print("\n")
        fp += len(pred_set - gold_set)
        fn += len(gold_set - pred_set)

        common_triples = gold_set.intersection(pred_set)
        true_positive_sum += len(common_triples)

    R = true_positive_sum / (true_positive_sum + fn)
    P = true_positive_sum / (true_positive_sum + fp)

    Fscore = 2 * P * R / (P + R)
    print(R)
    print(P)
    print(Fscore)


gold_lines = []
with open(gold_file, "r", encoding="utf8") as fr:
    for line in fr:
        e = line.strip()
        if len(e) > 0 and e[-1] == ".":
            gold_lines.append(e[:-1])
        else:
            gold_lines.append(e)

all_lines = []
with open(prediction_file, "r", encoding="utf8") as fr:
    for line in fr:
        line = line.replace(" [PAD]", "")
        e = line.strip()
        if len(e) > 0 and e[-1] == ".":
            all_lines.append(e[:-1])
        else:
            all_lines.append(e)

preds = []
golden = []
cnt = 0
fail_cnt = 0

for i, line in enumerate(zip(gold_lines, all_lines)):
    cnt += 1
    ret = []
    gold = []
    gold_sentences = split_sentence(line[0])
    pred_sentences = split_sentence(line[1])
    for sen in gold_sentences:
        gold_ans = convert_relis_sentence(sen)
        if gold_ans is not None:
            gold.append(gold_ans)
    golden.append(gold)

    for pred_Sen in pred_sentences:
        pred_ans = convert_relis_sentence(pred_Sen)
        if pred_ans is not None:
            ret.append(pred_ans)
    preds.append(ret)

do_eval(preds, golden)
