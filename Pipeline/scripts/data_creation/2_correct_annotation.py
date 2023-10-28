import os

from os.path import basename, splitext
from typing import List
import re

# nltk.download("stopwords")
from nltk.corpus import stopwords


def extract_indices_from_brat_annotation(indices: str) -> List[int]:
    indices = re.findall(r"\d+", indices)
    indices = sorted([int(i) for i in indices])
    return indices


def _get_mention_from_text(text, indices):
    tokens = []
    for i in range(0, len(indices), 2):
        start = int(indices[i])
        end = int(indices[i + 1])
        tokens.append(text[start:end])
    return " ".join(tokens)

# This folder contains raw documents (combined)
ann_path = "/Users/shashankgupta/Documents/Raredis/dai_et_al/test/input_ann"

# This folder contains .txt files separated
text_path = "/Users/shashankgupta/Documents/Raredis/dai_et_al/test/input_text/"

# output path to modified .txt files
output_path = "/Users/shashankgupta/Desktop/BioGPT_inputTest/Raredis_corrected_annotation/latest/test/input_text/"
text_list = os.listdir(text_path)

stop_words = stopwords.words("english")

total_entities = 0
entities_span_formated = 0

total_relations = 0
disjoint_entites = 0
disjoint_entites_2 = 0
relations_formated = 0
mentions_update = 0

idx = 0

for document in os.listdir(ann_path):
    doc_type = splitext(basename(document))[1]

    if doc_type != ".ann":
        continue

    entity_count = 0
    text = ""
    ent_list = []

    with open(os.path.join(ann_path, document), "r") as in_f:
        idx += 1
        doc_type = splitext(basename(document))[1]
        doc_name = splitext(basename(document))[0]
        print(idx, " doc name: ", document)
        # if doc_name != "Melkersson-Rosenthal-Syndrome":
        #     continue
        if doc_type == ".ann":
            document = document.replace(".ann", "")
            if document + ".txt" in text_list:
                txt = text_path + document + ".txt"
                with open(txt, "r+") as myf:
                    text = myf.read()
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

                    # write to output file
                    output_file = os.path.join(output_path, document + ".txt")
                    with open(output_file, "w") as out_f:
                        out_f.write(text)

            with open(os.path.join(
                    "/Users/shashankgupta/Desktop/BioGPT_inputTest/Raredis_corrected_annotation/latest/test"
                    "/ann", "%s.ann" % document),
                    'w') as fa:

                for line in in_f:
                    line = line.replace("\u2019", "'")
                    line = line.replace("\u201c", '"')
                    line = line.replace("\u201d", '"')
                    line = line.replace("\u2013", '-')
                    line = line.replace("\u2018", "'")
                    line = line.replace("\u00ed", "i")
                    line = line.replace("\u00f6", "o")
                    line = line.replace("\u00e9", "e")
                    line = line.replace("\u00e7", "c")
                    line = line.replace("\u00a0", " ")
                    line = line.replace("\u00e8", "e")
                    line = line.replace("\u2014", '-')
                    line = line.replace("\u00ba", ' ')
                    line = line.replace("\u03b2", 'B')

                    if line[0] == "T":
                        ent_change_flag = 0
                        entity_count += 1

                        # To correct the order of the offsets
                        sp = line.strip().split("\t")
                        assert len(sp) == 3
                        ent_id = sp[0].strip()
                        mention = sp[2]
                        sp = sp[1].split(" ")
                        label = sp[0].strip()
                        indices = extract_indices_from_brat_annotation(" ".join(sp[1:]))
                        if len(indices) > 2:
                            disjoint_entites_2 += 1
                        mention_from_text = _get_mention_from_text(text, indices)

                        if mention != mention_from_text:
                            print("Gold Entity", mention, " will be changed as taken from text ", mention_from_text)

                        second_tab = line.rfind('\t')
                        if ';' in line[:second_tab]:
                            if mention != mention_from_text:
                                print("Updating the mention from (%s) to (%s)." % (mention, mention_from_text))
                                formatted_indice = ";".join(
                                    ["{} {}".format(indices[i], indices[i + 1]) for i in range(0, len(indices), 2)])
                                line_temp = ent_id + "\t" + label + " " + formatted_indice + "\t" + mention_from_text + "\n"
                                fa.write(line_temp)
                                mentions_update += 1
                            else:
                                fa.write(line)
                            ent_list.append(int(line[1:line.find('\t')].strip()))
                            disjoint_entites = disjoint_entites + 1

                        else:
                            temp3 = mention_from_text.split()

                            # End character check
                            new_end_char = int(indices[-1])
                            char_at_new_end = text[new_end_char]

                            if char_at_new_end.isalnum():
                                old_ent = mention_from_text
                                new_end_token = temp3[-1] + char_at_new_end

                                if new_end_token in stop_words:
                                    new_ent = " ".join(temp3[0:-1])
                                    new_end_char -= 2
                                    temp3.pop()
                                else:
                                    temp3[-1] = new_end_token
                                    new_ent = " ".join(temp3)
                                    new_end_char += 1

                                if len(new_ent) == 0:
                                    continue

                                ent_change_flag = 1
                                indices[-1] = str(new_end_char)
                                print(f'(Entity: {ent_id}) is changed from Entity "{old_ent}" to "{new_ent}" for '
                                      f'document {doc_name}.')

                            # Start character check
                            if int(indices[0]) != 0:
                                new_start_char = int(indices[0]) - 1
                                char_at_new_start = text[new_start_char]

                                if char_at_new_start.isalnum():
                                    ent_change_flag = 1
                                    old_ent = mention_from_text
                                    new_start_token = char_at_new_start + temp3[0]

                                    if new_start_token in stop_words:
                                        new_ent = " ".join(temp3[1:])
                                        new_start_char += 3
                                        temp3.pop(0)
                                    elif text[new_start_char - 1] != " ":
                                        temp_tkn = temp3[0]
                                        len_temp_tkn = len(temp_tkn)
                                        temp3.pop(0)
                                        new_start_char += len_temp_tkn + 2
                                    else:
                                        temp3[0] = new_start_token
                                        new_ent = " ".join(temp3)
                                        # new_start_char -= 1

                                    new_ent = " ".join(temp3)
                                    if len(new_ent) == 0:
                                        continue
                                    indices[0] = str(new_start_char)
                                    print(
                                        f'(Entity: {ent_id}) is changed from Entity "{old_ent}" to "{new_ent}" for '
                                        f'document {doc_name}.')

                            ent_list.append(int(ent_id[1:]))
                            if ent_change_flag == 1:
                                entities_span_formated += 1
                                line1 = ent_id + "\t" + label + " " + str(indices[0]) + " " + str(
                                    indices[1]) + "\t" + " ".join(temp3) + "\n"
                                fa.write(line1)
                            else:
                                line4 = ent_id + "\t" + label + " " + str(indices[0]) + " " + str(
                                    indices[1]) + "\t" + mention_from_text + "\n"
                                fa.write(line4)


                    elif line[0] == 'R':
                        total_relations = total_relations + 1
                        spr = line.split()
                        spr1 = line.split("\t")
                        spr2 = spr1[1].split()
                        entity1 = spr2[1]
                        entity1_id = int(entity1[6:])

                        entity2 = spr2[2]
                        entity2_id = int(entity2[6:])

                        if entity1_id > entity_count or entity2_id > entity_count:
                            last_digit_zero = 0
                            relations_formated += 1
                            if entity1_id > entity_count and int((str(entity1_id))[-1]) == 0:
                                last_digit_zero = 1
                                new_entity1_id = spr2[1][:-1]
                                spr2[1] = new_entity1_id
                                entity1_id = int(new_entity1_id[6:])

                            if entity2_id > entity_count and int((str(entity2_id))[-1]) == 0:
                                last_digit_zero = 1
                                new_entity2_id = spr2[2][:-1]
                                spr2[2] = new_entity2_id
                                entity2_id = int(new_entity2_id[6:])

                        if entity1_id not in ent_list or entity2_id not in ent_list:
                            print(
                                f'Entity: (T{entity1_id}) or  (T{entity2_id}) is not present in the annotation of '
                                f'relation number {spr[0]} for document {doc_name}.')
                            continue

                        elif entity1_id == entity2_id:
                            print(
                                f'(Entity: T{entity1_id}) and  T{entity2_id} are same for '
                                f'relation number {spr[0]} for document {doc_name}.')
                            continue

                        new1 = " ".join(spr2)
                        spr1[1] = new1
                        fa.write("\t".join(spr1))

            total_entities += entity_count

print(total_entities, entities_span_formated, (entities_span_formated / total_entities) * 100)
print(total_relations, relations_formated, (relations_formated / total_relations) * 100)
