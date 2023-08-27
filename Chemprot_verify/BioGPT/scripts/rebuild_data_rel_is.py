import os
import json

data_dir = "/Users/shashankgupta/Downloads/Chemprot/BioGPT/"


def sort_triples(triples, text):
    sorted_triples = sorted(triples, key=lambda x: text.find(x['rel']))
    return sorted_triples


def build_target_seq_relis(triples):
    answer = ""
    for z in triples:
        rel = z["rel"].lower()
        if rel == "no relation":
            answer = "no relations; "

        else:
            arg1 = z["arg1"].lower()
            arg2 = z["arg2"].lower()
            rel = z["rel"].lower()

            if rel == "cpr:3":
                answer += f"the relationship between {arg1} and {arg2} is activator; "

            elif rel == "cpr:4":
                answer += f"the relationship between {arg1} and {arg2} is inhibitor; "

            elif rel == "cpr:5":
                answer += f"the relationship between {arg1} and {arg2} is agonist; "

            elif rel == "cpr:6":
                answer += f"the relationship between {arg1} and {arg2} is antagonist; "

            elif rel == "cpr:9":
                answer += f"the relationship between {arg1} and {arg2} is substrate; "

    return answer[:-2] + "."


def loader(fname, fn):
    ret = []
    null_cnt = 0
    suc_cnt = 0
    null_flag = False
    with open(fname, "r", encoding="utf8") as fr:
        data = json.load(fr)
    for pmid, v in data.items():
        content = v["abstract"].strip().replace('\n', ' ')

        content = content.lower()
        if v["triples"] is None or len(v["triples"]) == 0:
            if not null_flag:
                print(f"Following PMID in {fname} has no extracted triples:")
                null_flag = True
            print(f"{pmid} ", end="")
            null_cnt += 1

        else:
            triples = v['triples']
            triples = sort_triples(triples, content)
            answer = fn(triples)
            ret.append((pmid, content, answer))
            suc_cnt += 1
    if null_flag:
        print("")
    print(f"{len(data)} samples in {fname} has been processed with {null_cnt} samples has no triples extracted.")
    return ret


def dumper(content_list, prefix):
    fw_pmid = open(prefix + ".pmid", "w")
    fw_content = open(prefix + ".x", "w")
    fw_label = open(prefix + ".y", "w")

    for pmid, x, y in content_list:
        print(pmid, file=fw_pmid)
        print(x, file=fw_content)
        print(y, file=fw_label)

    fw_pmid.close()
    fw_content.close()
    fw_label.close()


def worker(fname, prefix, fn):
    ret = loader(fname, fn)
    dumper(ret, prefix)


for split in ['train', 'valid', 'test']:
    worker(os.path.join(f"{data_dir}", f"{split}.json"), os.path.join(f"{data_dir}", f"relis_{split}"),
           build_target_seq_relis)
