import json, os
import pandas as pd
import re

true_positive_sum, pred_sum, true_sum = 0, 0, 0

prediction_file = "/Users/temp1150_best/test_beam.txt"
gold_file = "/Users/temp1150_best/test_gold.txt"


def split_sentence(line):
    sentences = re.split(r"; ", line)
    return list(set(sentences))


def convert_relis_sentence(sentence):
    ans = None
    segs = re.match(r"the relationship between (.*) and (.*) is (.*)", sentence)
    if segs is not None:
        segs = segs.groups()

        relation = segs[2].strip()
        rel = ''
        if relation == "activator":
            rel = "cpr:3"
        elif relation == "inhibitor":
            rel = "cpr:4"
        elif relation == "agonist":
            rel = "cpr:5"
        elif relation == "antagonist":
            rel = "cpr:6"
        elif relation == "substrate":
            rel = "cpr:9"

        ans = (segs[0].strip(), rel, segs[1].strip())

    elif sentence == "no relations":
        ans = ("", "no relation", "")
    return ans


def do_eval(preds, golden):
    num_missing = 0
    fp = 0
    fn = 0
    tn = 0

    cpr3_fp, cpr3_tp, cpr3_fn = 0, 0, 0
    cpr4_fp, cpr4_tp, cpr4_fn = 0, 0, 0
    cpr5_fp, cpr5_tp, cpr5_fn = 0, 0, 0
    cpr6_fp, cpr6_tp, cpr6_fn = 0, 0, 0
    cpr9_fp, cpr9_tp, cpr9_fn = 0, 0, 0

    columns = ['doc_name', 'gold_rels', 'pred_rels', 'rel_type', 'fp_prediction', "fn_prediction"]
    df = pd.DataFrame(columns=columns)
    idx = 0

    for gold, pred in zip(golden, preds):
        idx += 1
        gold_arg1_set, gold_arg2_set, gold_rel_set, gold_set = set(), set(), set(), set()
        pred_arg1_set, pred_arg2_set, pred_rel_set, pred_set = set(), set(), set(), set()
        gold_rel = ""
        for tp in gold:
            gold_rel = tp[1].strip().lower()
            if gold_rel != "no relation":
                arg1 = tp[0].strip().lower()
                arg2 = tp[2].strip().lower()
                gold_arg1_set.add(arg1)
                gold_arg2_set.add(arg2)
                gold_rel_set.add(gold_rel)
                gold_set.add((arg1, arg2, gold_rel))

        if pred:
            for p_tp in pred:
                rel = p_tp[1].strip().lower()
                if rel == "no relation" and gold_rel == "no relation":
                    tn += 1
                    continue

                elif gold_rel == "no relation" and rel != "no relation":
                    fp += len(pred_set)
                    continue

                elif gold_rel != "no relation" and rel == "no relation":
                    fn += len(gold_set)
                    continue

                arg1 = p_tp[0].strip().lower()
                arg2 = p_tp[2].strip().lower()
                pred_arg1_set.add(arg1)
                pred_arg2_set.add(arg2)
                pred_rel_set.add(rel)
                pred_set.add((arg1, arg2, rel))

        fp_rel = pred_rel_set - gold_rel_set

        new_rows = [
            {'doc_name': idx, 'gold_rels': gold_rel_set, 'pred_rels': pred_rel_set, 'rel_type': "", 'fp_prediction': "",
             'fn_prediction': ""}]
        df = df.append(new_rows, ignore_index=True)

        fp_dic = dict()
        for x in fp_rel:
            fp_lst = set()
            for y in pred_set:
                if y[2] == x:
                    fp_lst.add(y)

            if x == "cpr:3":
                cpr3_fp += len(fp_lst)
            elif x == "cpr:4":
                cpr4_fp += len(fp_lst)
            elif x == "cpr:5":
                cpr5_fp += len(fp_lst)
            elif x == "cpr:6":
                cpr6_fp += len(fp_lst)
            elif x == "cpr:9":
                cpr9_fp += len(fp_lst)

            fp_dic[x] = fp_lst

        fn_dic = dict()
        fp_dic_2 = dict()

        for z in gold_rel_set:
            gold = set()
            prediction = set()
            for f in gold_set:
                if f[2] == z:
                    gold.add(f)
            for g in pred_set:
                if g[2] == z:
                    prediction.add(g)

            fn_rel = gold - prediction
            tp_rel = gold.intersection(prediction)
            fp_rels = prediction - gold

            if len(fn_rel) or len(tp_rel) or len(fp_rels) > 0:
                if z == "cpr:3":
                    cpr3_fn += len(fn_rel)
                    cpr3_tp += len(tp_rel)
                    cpr3_fp += len(fp_rels)
                elif z == "cpr:4":
                    cpr4_fn += len(fn_rel)
                    cpr4_tp += len(tp_rel)
                    cpr4_fp += len(fp_rels)
                elif z == "cpr:5":
                    cpr5_fn += len(fn_rel)
                    cpr5_tp += len(tp_rel)
                    cpr5_fp += len(fp_rels)
                elif z == "cpr:6":
                    cpr6_fn += len(fn_rel)
                    cpr6_tp += len(tp_rel)
                    cpr6_fp += len(fp_rels)
                elif z == "cpr:9":
                    cpr9_fn += len(fn_rel)
                    cpr9_tp += len(tp_rel)
                    cpr9_fp += len(fp_rels)

                fn_dic[z] = fn_rel
                fp_dic_2[z] = fp_rels

        if len(fp_dic) > 0:
            for a in fp_dic.keys():
                new_rows = [
                    {'doc_name': "", 'gold_rels': "", 'pred_rels': "", 'rel_type': a, 'fp_prediction': fp_dic[a],
                     'fn_prediction': ""}]
                df = df.append(new_rows, ignore_index=True)

        if len(fn_dic) > 0 or len(fp_dic_2) > 0:
            for b in fn_dic.keys():
                new_rows = [
                    {'doc_name': "", 'gold_rels': "", 'pred_rels': "", 'rel_type': b, 'fp_prediction': fp_dic_2[b],
                     'fn_prediction': fn_dic[b]}]
                df = df.append(new_rows, ignore_index=True)

    # Produce Scores
    P_cpr3 = cpr3_tp / (cpr3_tp + cpr3_fp)
    R_cpr3 = cpr3_tp / (cpr3_tp + cpr3_fn)
    Fscore_cpr3 = 2 * P_cpr3 * R_cpr3 / (P_cpr3 + R_cpr3)

    print("cpr3 precision is: ", P_cpr3)
    print("cpr3 Recall is: ", R_cpr3)
    print("cpr3 F-score is: ", Fscore_cpr3)

    # is_a Scores
    P_cpr4 = cpr4_tp / (cpr4_tp + cpr4_fp)
    R_cpr4 = cpr4_tp / (cpr4_tp + cpr4_fn)
    Fscore_cpr4 = 2 * P_cpr4 * R_cpr4 / (P_cpr4 + R_cpr4)

    print("cpr4 precision is: ", P_cpr4)
    print("cpr4 Recall is: ", R_cpr4)
    print("cpr4 F-score is: ", Fscore_cpr4)

    # is_acron Scores
    P_cpr9 = cpr9_tp / (cpr9_tp + cpr9_fp)
    R_cpr9 = cpr9_tp / (cpr9_tp + cpr9_fn)
    Fscore_cpr9 = 2 * P_cpr9 * R_cpr9 / (P_cpr9 + R_cpr9)

    print("cpr9 precision is: ", P_cpr9)
    print("cpr9 Recall is: ", R_cpr9)
    print("cpr9 F-score is: ", Fscore_cpr9)

    # is_synon Scores
    P_cpr6 = cpr6_tp / (cpr6_tp + cpr6_fp)
    R_cpr6 = cpr6_tp / (cpr6_tp + cpr6_fn)
    Fscore_cpr6 = 2 * P_cpr6 * R_cpr6 / (P_cpr6 + R_cpr6)

    print("cpr6 precision is: ", P_cpr6)
    print("cpr6 Recall is: ", R_cpr6)
    print("cpr6 F-score is: ", Fscore_cpr6)

    # increase_risk_of Scores
    P_cpr5 = cpr5_tp / (cpr5_tp + cpr5_fp)
    R_cpr5 = cpr5_tp / (cpr5_tp + cpr5_fn)
    Fscore_cpr5 = 2 * P_cpr5 * R_cpr5 / (P_cpr5 + R_cpr5)

    print("cpr5 precision is: ", P_cpr5)
    print("cpr5 Recall is: ", R_cpr5)
    print("cpr5 F-score is: ", Fscore_cpr5)

    # Overall
    fp = cpr3_fp + cpr4_fp + cpr5_fp + cpr6_fp + cpr9_fp
    fn += cpr3_fn + cpr4_fn + cpr5_fn + cpr6_fn + cpr9_fn
    tp = cpr3_tp + cpr4_tp + cpr5_tp + cpr6_tp + cpr9_tp

    P_overall = tp / (tp + fp)
    R_overall = tp / (tp + fn)
    F_overall = 2 * P_overall * R_overall / (P_overall + R_overall)

    print("Overall precision is: ", P_overall)
    print("Overall Recall is: ", R_overall)
    print("Overall F-score is: ", F_overall)

    #
    # df.to_csv("/Users/shashankgupta/Downloads/Chemprot/BioMedLM/results/temp1000/error_analysis_rel_type.csv", index=False)


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
