import json
import os
from os.path import splitext, basename

# path to ann folder
ann_path = "/home/valid/ann"

# path to txt folder
text_path = "/home/valid/input_text"
final_doc = dict()
null_cnt = 0

with open('/home/without_token_copy_instruction/with_ent_type/valid.json', 'w') as outfile:
    for document in os.listdir(ann_path):
        relation_present = 0

        with open(os.path.join(ann_path, document), "r") as in_f:
            doc_type = splitext(basename(document))[1]
            if doc_type == ".ann":
                document = document.replace(".ann", "")
                text = open(os.path.join(text_path, "%s.txt" % document)).read()
                text = text.replace("\u2019", "'")
                text = text.replace("\u201c", '"')
                text = text.replace("\u201d", '"')
                text = text.replace("\u2013", '-')
                text = text.replace("\u2018", "'")
                text = text.replace("\u00ed", "i")
                text = text.replace("\u00f6", "o")
                text = text.replace("\u00e9", "e")
                text = text.replace("\u00e7", "c")
                text = text.replace("\u00a0", " ")
                text = text.replace("\u00e8", "e")
                text = text.replace("\u2014", '-')
                text = text.replace("\u00ba", ' ')
                text = text.replace("\u03b2", 'B')

                content = text.strip().replace('\n', ' ')
                content = content.lower()

                # comment one of the below to include copy instruction or not

                # content = "consider the abstract: $ " + content + " $ from the given abstract, find all the entities and relations among them. do not generate any token outside the abstract."  # copy instruction
                content = "consider the abstract: $ " + content + " $ from the given abstract, find all the entities and relations among them."  # without copy instruction

                ent_dict = {}
                triples = []

                for line in in_f:
                    if line[0] == "T":
                        ent_data = line.split("\t")
                        ent_type = ent_data[1].split()[0]
                        mention = ent_data[-1].strip().replace('\n', '')

                        if ent_type == "ANAPHOR":
                            mention = "\"" + mention + "\""

                        # comment one of the below to include entity type or not

                        # ent_dict[ent_data[0]] = mention.lower() # without entity type
                        ent_dict[ent_data[0]] = (ent_type + " " + mention).lower()  # include entity type

                    elif line[0] == "R":
                        temp = dict()
                        rel_data = line.split("\t")
                        relation_present = 1
                        relation_type = rel_data[1].split()[0]
                        rel_arg_1 = rel_data[1].split()[1][5:]
                        rel_arg_2 = rel_data[1].split()[2][5:]
                        temp["rel"] = relation_type.lower()

                        if rel_arg_1 in ent_dict.keys():
                            temp["arg1"] = ent_dict[rel_arg_1]
                        else:
                            print(f"Entity ID {rel_arg_1} does not exist in the annotation file of {document}")
                            continue

                        if rel_arg_2 in ent_dict.keys():
                            temp["arg2"] = ent_dict[rel_arg_2]
                        else:
                            print(f"Entity ID {rel_arg_2} does not exist in the annotation file of {document}")
                            continue

                        triples.append(temp)

                if relation_present == 0:
                    triples.append({"rel": "no relation"})
                    print(f"Following document {document} has no relations: ")
                    final_doc[document] = {"abstract": content}
                    final_doc[document]["triples"] = triples
                    final_doc[document]["pmid"] = document
                    null_cnt += 1

                else:
                    final_doc[document] = {"abstract": content}
                    final_doc[document]["triples"] = triples
                    final_doc[document]["pmid"] = document

    json.dump(final_doc, outfile)
