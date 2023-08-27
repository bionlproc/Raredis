import json, os
import pandas as pd

true_positive_sum, pred_sum, true_sum = 0, 0, 0

pred_file = "/Users/shashankgupta/Downloads/Chemprot/BioGPT/generate_checkpoint_last.pt.detok.extracted.json"
gold_file = "/Users/shashankgupta/Downloads/Chemprot/BioGPT/test.json"
pmids_file = "/Users/shashankgupta/Downloads/Chemprot/BioGPT/data/relis_test.pmid"


def do_eval(preds, pmids, golden):
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

    for pred, idx in zip(preds, pmids):
        print(idx)
        gold_arg1_set, gold_arg2_set, gold_rel_set, gold_set = set(), set(), set(), set()
        pred_arg1_set, pred_arg2_set, pred_rel_set, pred_set = set(), set(), set(), set()
        gold_rel = ""
        if idx not in golden:
            num_missing += 1
            # print("----Missing:", idx)
            continue
        if golden[idx]["triples"]:
            for tp in golden[idx]["triples"]:
                gold_rel = tp["rel"].strip().lower()
                if gold_rel != "no relation":
                    arg1 = tp["arg1"].strip().lower()
                    arg2 = tp["arg2"].strip().lower()
                    gold_arg1_set.add(arg1)
                    gold_arg2_set.add(arg2)
                    gold_rel_set.add(gold_rel)
                    gold_set.add((arg1, arg2, gold_rel))

        if pred["triple_list_pred"] and pred["triple_list_pred"][0]["subject"] != 'failed':
            for tp in pred["triple_list_pred"]:
                rel = tp["relation"].strip().lower()
                if rel == "no relation" and gold_rel == "no relation":
                    tn += 1
                    continue

                elif gold_rel == "no relation" and rel != "no relation":
                    fp += len(pred_set)
                    continue

                elif gold_rel != "no relation" and rel == "no relation":
                    fn += len(gold_set)
                    continue

                arg1 = tp["subject"].strip().lower()
                arg2 = tp["object"].strip().lower()
                pred_arg1_set.add(arg1)
                pred_arg2_set.add(arg2)
                pred_rel_set.add(rel)
                pred_set.add((arg1, arg2, rel))

        fp_rel = pred_rel_set - gold_rel_set

        new_rows = [{'doc_name': idx, 'gold_rels': gold_rel_set, 'pred_rels': pred_rel_set, 'rel_type': "", 'fp_prediction': "", 'fn_prediction': ""}]
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

            if len(fn_rel) or len(tp_rel) or len(fp_rels)> 0:
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
                new_rows = [{'doc_name': "", 'gold_rels': "", 'pred_rels': "", 'rel_type': a, 'fp_prediction': fp_dic[a], 'fn_prediction': ""}]
                df = df.append(new_rows, ignore_index=True)

        if len(fn_dic) > 0 or len(fp_dic_2) > 0:
            for b in fn_dic.keys():
                new_rows = [{'doc_name': "", 'gold_rels': "", 'pred_rels': "", 'rel_type': b, 'fp_prediction': fp_dic_2[b], 'fn_prediction': fn_dic[b]}]
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
    fp = cpr3_fp + cpr4_fp+ cpr5_fp + cpr6_fp + cpr9_fp
    fn += cpr3_fn + cpr4_fn + cpr5_fn + cpr6_fn + cpr9_fn
    tp = cpr3_tp + cpr4_tp + cpr5_tp + cpr6_tp + cpr9_tp

    P_overall = tp / (tp+fp)
    R_overall = tp / ( tp +fn)
    F_overall = 2*P_overall*R_overall/ (P_overall + R_overall)

    print("Overall precision is: ", P_overall)
    print("Overall Recall is: ", R_overall)
    print("Overall F-score is: ", F_overall)

    #
    # df.to_csv(
    #     "/Users/Downloads/Chemprot/BioGPT/with_ent_type/rel-is/error_analysis_rel_type_last.csv", index=False)


preds = []
with open(pred_file) as reader:
    for line in reader:
        preds.append(json.loads(line))

with open(gold_file) as reader:
    golden = json.load(reader)

with open(pmids_file) as reader:
    if '.json' in pmids_file:
        pmids = json.load(reader)
    else:
        pmids = []
        for line in reader:
            pmids.append(line.strip())

print("\n====File: ", os.path.basename(pred_file))
do_eval(preds, pmids, golden)

