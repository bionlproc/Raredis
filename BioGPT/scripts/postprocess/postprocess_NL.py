'''
This file post processes the detok output from infer.sh and dumps the json file for  evaluation
'''


import os
import sys
import re
import json

# path to .detok file generated from infer.sh
out_file = "/home/generate_checkpoint_avg.pt.detok"

prefix = [
    '(learned[0-9]+ )+',
    'we can conclude that',
    'we have that',
    'in conclusion,',
]


def strip_prefix(line):
    for p in prefix:
        res = re.search(p, line)
        if res is not None:
            line = re.split(p, line)[-1].strip()
            break
    return line


def split_sentence(line):
    sentences = re.split(r"; ", line)
    return list(set(sentences))

# This method replaces the template ent type to original ent type
def get_original_ent_type(enttype):
    enttype = enttype.replace("rare disease", "raredisease")
    enttype = enttype.replace("rare skin disease", "skinraredisease")
    enttype = enttype.replace("signs", "sign")

    return enttype

# This method extracts the objection, subject and the predicate with ent type
def convert_relis_sentence(sentence):
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

    elif sentence.strip() == "no relations present in the abstract":
        ans = ("", "no relation", "")

    return ans

# This method extracts the objection, subject and the predicate without ent type

def convert_relis_sentence_wo_ent_type(sentence):
    sentence = get_original_ent_type(sentence)
    ans = None
    produces = re.match(r"^(.*) produces (.*)", sentence)
    synonym = re.match(r"the (.*) and the (.*) are synonyms", sentence)
    anaphora = re.match(r"the term (.*?) refers back to the entity (.*)", sentence)
    acronym = re.match(r"the acronym (.*) stands for (.*)", sentence)
    increases_risk_of = re.match(r"the presence of the (.*?) increases the risk of developing the (.*)", sentence)
    is_a = re.match(r"the (.*) is a type of (.*)", sentence)
    if produces is not None:
        produces = produces.groups()
        ans = (produces[0].strip(), "produces", produces[1].strip())
    elif synonym is not None:
        synonym = synonym.groups()
        ans = (synonym[0].strip(), "is_synon", synonym[1].strip())
    elif anaphora is not None:
        anaphora = anaphora.groups()
        ans = (anaphora[1].strip(), "anaphora", anaphora[0].strip("'"))
    elif acronym is not None:
        acronym = acronym.groups()
        ans = (acronym[0].strip(), "is_acron", acronym[1].strip())
    elif increases_risk_of is not None:
        increases_risk_of = increases_risk_of.groups()
        ans = (increases_risk_of[0].strip(), "increases_risk_of", increases_risk_of[1].strip())
    elif is_a is not None:
        is_a = is_a.groups()
        ans = (is_a[0].strip(), "is_a", is_a[1].strip())

    elif sentence.strip() == "no relations present in the abstract":
        ans = ("", "no relation", "")

    return ans

def converter(sample, h_idx=0, r_idx=1, t_idx=2):
    ret = {"triple_list_gold": [], "triple_list_pred": [], "new": [], "lack": [], "id": [0]}
    for s in sample:
        ret["triple_list_pred"].append({"subject": s[h_idx], "relation": s[r_idx], "object": s[t_idx]})
    return ret


all_lines = []
with open(out_file, "r", encoding="utf8") as fr:
    for line in fr:
        e = line.strip()
        if len(e) > 0 and e[-1] == ".":
            all_lines.append(e[:-1])
        else:
            all_lines.append(e)

hypothesis = []
cnt = 0
fail_cnt = 0

for i, line in enumerate(all_lines):
    cnt += 1
    ret = []
    strip_line = strip_prefix(line)
    sentences = split_sentence(strip_line)
    for sen in sentences:
        ans = convert_relis_sentence(sen)   # Change method name here depending on if you want to extract entities with ent type or w/o ent type
        if ans is not None:
            ret.append(ans)
    if len(ret) > 0:
        hypothesis.append(ret)
    else:
        hypothesis.append([("failed", "failed", "failed")])
        fail_cnt += 1
        print("Failed:id:{}, line:{}".format(i + 1, line))

ret_formatted = []
for i in range(len(hypothesis)):
    ret_formatted.append(converter(hypothesis[i]))

with open(f"{out_file}.extracted.json", "w", encoding="utf8") as fw:
    for eg in ret_formatted:
        print(json.dumps(eg), file=fw)

print(f"failed = {fail_cnt}, total = {cnt}")
