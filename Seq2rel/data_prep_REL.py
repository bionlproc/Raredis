from glob import glob
from os import listdir
from os.path import splitext, basename

import nltk
import logging

nltk.download("stopwords")
from nltk.corpus import stopwords

stop_words = stopwords.words("english")
total_relations = 0
total_entities = 0
total_rel_annotation_errors = 0
total_rel_annotation_errors_fixed = 0
total_rel_annotation_errors_unfixed = 0
final_output_string = ""
dump_file = ""
counter = 0
logging.basicConfig(filename="home/validWarning.log", level=logging.DEBUG)


def get_paired_files(all_files):
    """
    Check that there is both a .txt and .ann file for each filename, and return
    a list of tuples of the form ("myfile.txt", "myfile.ann"). Triggers an
    excpetion if one of the two files is missing, ignores any files that don't
    have either a .txt or .ann extension.
    parameters:
        all_files, list of str: list of all filenames in directory
    returns:
        paired_files, list of tuple: list of file pairs
    """
    paired_files = []

    # Get a set of all filenames without extensions
    basenames = set([splitext(name)[0] for name in all_files])

    # Check that there are two files with the right extenstions and put in list
    for name in basenames:

        # Get files with the same name
        matching_filenames = glob(f"{name}.*")

        # Check that both .txt and .ann are present
        txt_present = True if f"{name}.txt" in matching_filenames else False
        ann_present = True if f"{name}.ann" in matching_filenames else False

        # Put in the list or raise an exception
        if txt_present and ann_present:
            paired_files.append((f"{name}.txt", f"{name}.ann"))
        elif txt_present and not ann_present:
            raise ValueError("The .ann file is missing "
                             f"for the basename {name}. Please fix or delete.")
        elif ann_present and not txt_present:
            raise ValueError("The .txt file is missing "
                             f"for the basename {name}. Please fix or delete.")

    return paired_files

# Move all the .txt files and .ann from their respective folders in one location
# For example, 600 files in .ann and 600 files in .txt, move them to one location so directory has 1200 files combined
data_directory = "/home/seq2rel/data/train"
all_files = [
    f"{data_directory}/{name}" for name in listdir(data_directory)
]

paired_files = get_paired_files(all_files)

for fname_pair in paired_files:
    ann = fname_pair[1]
    txt = fname_pair[0]
    counter += 1
    output = ""

    doc_name = splitext(basename(fname_pair[1]))[0]
    logging.info((counter, " Document Processing: ", doc_name))

    with open(txt) as myf:
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
        text = text.replace("\n", "")

    with open(ann) as myf:
        lines = myf.readlines()
        #
        for i, x in enumerate(lines):
            x = x.replace("\u2019", "'")
            x = x.replace("\u201c", '"')
            x = x.replace("\u201d", '"')
            x = x.replace("\u2013", '-')
            x = x.replace("\u2018", "'")
            x = x.replace("\u00ed", "i")
            x = x.replace("\u00f6", "o")
            x = x.replace("\u00e9", "e")
            x = x.replace("\u00e7", "c")
            x = x.replace("\u00a0", " ")
            x = x.replace("\u00e8", "e")
            x = x.replace("\u2014", '-')
            x = x.replace("\u00ba", ' ')
            x = x.replace("\u03b2", 'B')
            lines[i] = x

    entities = {}
    relations = {}
    entity_count = 0
    parts = []
    dic_entity = {}

    for annotation in lines:
        if annotation[0] == "T":
            parts = annotation.split("\t")
        else:
            parts = annotation.split()

        if parts[0].startswith("T"):
            entity_count += 1
            total_entities += 1

            entity_type = (parts[1].split())[0]
            entity_name = parts[2][:-1]

            if entity_type not in dic_entity.keys():
                dic_entity[entity_type] = [entity_name]

            else:
                lst = dic_entity.get(entity_type)
                lst.append(entity_name)
                dic_entity[entity_type] = lst

            entities[parts[0]] = entity_name + " @" + entity_type + "@"
            # entities[parts[0]] = entity_name

        elif parts[0].startswith("R"):
            total_relations = total_relations + 1

            entity1 = parts[2]
            entity1_id = int(entity1[6:])

            entity2 = parts[3]
            entity2_id = int(entity2[6:])

            if entity1_id > entity_count or entity2_id > entity_count:
                total_rel_annotation_errors += 1

                last_digit_zero = 0
                if entity1_id > entity_count and int((str(entity1_id))[-1]) == 0:
                    last_digit_zero = 1
                    new_entity1_id = parts[2][:-1]
                    parts[2] = new_entity1_id
                    total_rel_annotation_errors_fixed += 1

                if entity2_id > entity_count and int((str(entity2_id))[-1]) == 0:
                    last_digit_zero = 1
                    new_entity2_id = parts[3][:-1]
                    parts[3] = new_entity2_id
                    total_rel_annotation_errors_fixed += 1

                if last_digit_zero == 1:
                    logging.warning(f'Relation {parts[0]} Annotation error. '
                                    f'Previous Annotation: {entity1}  {entity2} '
                                    f'New Annotation {parts[2]}  {parts[3]}')

                else:
                    logging.warning(f'Relation {parts[0]} will be missed because of Annotation error. '
                                    f'One of the arguments: {entity1}  {entity2} do not exist.')

                    total_rel_annotation_errors_unfixed += 1

                    continue

            relations[parts[0]] = parts[1] + " " + " ".join(parts[2:])

    if relations:
        for relation in relations.values():
            relation_parts = relation.split()
            relation_type = relation_parts[0]
            entity1_parts = relation_parts[1].split(":")
            entity2_parts = relation_parts[2].split(":")

            if entity1_parts[1] in entities.keys():
                entity1 = entities[entity1_parts[1]]
            else:
                logging.warning((entity1_parts[1], " is not found in annotation file "))
                continue

            if entity2_parts[1] in entities.keys():
                entity2 = entities[entity2_parts[1]]
            else:
                logging.warning((entity1_parts[1], " is not found in annotation file "))
                continue

            output += f"{entity1} {entity2} @{relation_type}@ "

    else:
        output += "@NOREL@"

    final_output_string = text + "\t" + output + "\n"
    dump_file = dump_file + final_output_string

print("Total Entities: ", total_entities)
print("Total Relations: ", total_relations)

# Write out doc dictionaries
with open("/home/seq2rel/data/train.txt", "w") as file:
    file.writelines(dump_file)
