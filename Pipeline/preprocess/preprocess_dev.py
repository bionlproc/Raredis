import json
import random

Path_train = '/Users/XA/Desktop/Raredis/Pipeline/raw_data/dev/dev.json'
Path_train_re = '/Users/XA/Desktop/Raredis/Pipeline/raw_data/dev/dev_re.json'
output_file_path = "/Users/XA/Desktop/Raredis/Pipeline/preprocessed_data/dev.json"

train = []
for line in open(Path_train, 'r'):
    train.append(json.loads(line))

train_re = []
for line in open(Path_train_re, 'r'):
    train_re.append(json.loads(line))

# Set up the format
for i in range(len(train)):
    train[i]['doc_key'] = train[i].pop('doc')
    train[i]['sentences'] = train[i].pop('tokens')
    train[i]['ner'] = train[i].pop('entities')
    train[i]['ner_modified'] = train[i]['ner'].copy()
    train[i]['sentences_modified'] = train[i]['sentences'].copy()
    train[i].pop('start')
    train[i].pop('end')
    train[i].pop('text')

def str2intlist(s):
    l = []
    l.append(int(s.split(',')[0]))
    l.append(int(s.split(',')[1]))
    return l

# Convert number strings to number lists for ner
for i in range(len(train)):
    l = []
    for j in range(len(train[i]['ner'])):
        l_temp = []
        for k in range(len(train[i]['ner'][j]['span'])):
            l_temp.append(str2intlist(train[i]['ner'][j]['span'][k]))
        l_temp.append(train[i]['ner'][j]['type'])
        l.append(l_temp)
    train[i]['ner'] = l

# Convert number strings to number lists for ner_modified
for i in range(len(train)):
    l = []
    for j in range(len(train[i]['ner_modified'])):
        l_temp = []
        for k in range(len(train[i]['ner_modified'][j]['span'])):
            l_temp.append(str2intlist(train[i]['ner_modified'][j]['span'][k]))
        l_temp.append(train[i]['ner_modified'][j]['type'])
        l.append(l_temp)
    train[i]['ner_modified'] = l

# Remove discontinuous entities with more than 2 fragments
for i in range(len(train)):
    indices_to_remove = set()
    for j in range(len(train[i]['ner_modified'])):
        if len(train[i]['ner_modified'][j]) >= 4:
            indices_to_remove.add(j)
    new_list = [sublist for idx, sublist in enumerate(train[i]['ner_modified']) if idx not in indices_to_remove]
    train[i]['ner_modified'] = new_list

##################################################################################
########### Check non-overlapped and overlapped discontinuous entities ###########
##################################################################################

# The number of non-overlapped fragments for the current entity
def num_nonoverlap_fragments_current_entity(ner_lists, current_entitiy_idx):
    num = 0
    if 0 < current_entitiy_idx < len(ner_lists) - 1:  # Neither the 1st nor the last entity
        for k in range(len(ner_lists[current_entitiy_idx]) - 1):  # -1 excludes entity types such as 'SIGN'
            cur_entity_cur_fragment_left_pos = ner_lists[current_entitiy_idx][k][0]
            cur_entity_cur_fragment_right_pos = ner_lists[current_entitiy_idx][k][1]
            previous_entity_last_fragment_right_pos = ner_lists[current_entitiy_idx - 1][-2][1]
            next_entity_first_fragment_left_pos = ner_lists[current_entitiy_idx + 1][0][0]
            if (
                    previous_entity_last_fragment_right_pos < cur_entity_cur_fragment_left_pos < next_entity_first_fragment_left_pos
                    and previous_entity_last_fragment_right_pos < cur_entity_cur_fragment_right_pos < next_entity_first_fragment_left_pos):
                num += 1
    elif current_entitiy_idx == 0:  # 1st entity
        for k in range(len(ner_lists[current_entitiy_idx]) - 1):
            cur_entity_cur_fragment_left_pos = ner_lists[current_entitiy_idx][k][0]
            cur_entity_cur_fragment_right_pos = ner_lists[current_entitiy_idx][k][1]
            next_entity_first_fragment_left_pos = ner_lists[current_entitiy_idx + 1][0][0]
            if (cur_entity_cur_fragment_left_pos < next_entity_first_fragment_left_pos
                    and cur_entity_cur_fragment_right_pos < next_entity_first_fragment_left_pos):
                num += 1
    else:  # Last entity
        for k in range(len(ner_lists[current_entitiy_idx]) - 1):
            cur_entity_cur_fragment_left_pos = ner_lists[current_entitiy_idx][k][0]
            cur_entity_cur_fragment_right_pos = ner_lists[current_entitiy_idx][k][1]
            previous_entity_last_fragment_right_pos = ner_lists[current_entitiy_idx - 1][-2][1]
            if (cur_entity_cur_fragment_left_pos > previous_entity_last_fragment_right_pos
                    and cur_entity_cur_fragment_right_pos > previous_entity_last_fragment_right_pos):
                num += 1
    return num

def check_cur_entity_overlap(ner_lists, current_entitiy_idx):
    if num_nonoverlap_fragments_current_entity(ner_lists, current_entitiy_idx) == len(ner_lists[current_entitiy_idx]) - 1:
        return False # All fragments of current entity do not overlap with other entities
    else:
        return True


num_non_overlapped = 0
num_overlapped = 0

for doc in range(len(train)):
    ner_lists = train[doc]['ner_modified']
    for current_entitiy_idx in range(len(ner_lists)):
        if len(ner_lists[current_entitiy_idx]) == 3:  # 2-fragment discontinuous entities
            num_nonoverlap_fragments = num_nonoverlap_fragments_current_entity(ner_lists, current_entitiy_idx)
            if not check_cur_entity_overlap(ner_lists, current_entitiy_idx):
                # print(f"The {current_entitiy_idx}th discontinuous entity in doc_{doc} is non-overlapped.")
                num_non_overlapped += 1
            else:
                # print(f"The {current_entitiy_idx}th discontinuous entity in doc_{doc} is overlapped.")
                num_overlapped += 1
# print(f'num_non_overlapped is {num_non_overlapped} and num_overlapped is {num_overlapped}')


# Check whether two fragments are overlapping
def check_overlap(fragment1, fragment2):
    # Extract the left and right positions of each fragment
    left1, right1 = fragment1
    left2, right2 = fragment2
    # Check for overlap
    if right1 < left2 or right2 < left1:
        return False  # Non-overlapping fragments
    else:
        return True

##############################################################################
########################## Rule 1 ############################################
##############################################################################

# Move the 2nd fragment to the right of the 1st fragment
def modify_tokens_rule_1(tokens, ner_lists, current_entitiy_idx):
    # All tokens before (including) the last token of the 1st fragment
    tokens_before_1st_fragment = tokens[:ner_lists[current_entitiy_idx][0][1] + 1]
    # All tokens of the 2nd fragment
    tokens_2nd_fragment = tokens[ner_lists[current_entitiy_idx][1][0]:ner_lists[current_entitiy_idx][1][1] + 1]
    # All tokens after (including) the 1st token of the 2nd fragment
    tokens_after_2nd_fragment = tokens[ner_lists[current_entitiy_idx][0][1] + 1:]
    new_tokens = tokens_before_1st_fragment + tokens_2nd_fragment + tokens_after_2nd_fragment
    return new_tokens


def modify_offsets_rule_1(ner_lists, current_entitiy_idx):
    # Number of tokens of the 2nd fragment
    len_2nd_fragment = ner_lists[current_entitiy_idx][1][1] - ner_lists[current_entitiy_idx][1][0] + 1
    last_token_pos_1st_fragment = ner_lists[current_entitiy_idx][0][1]
    # Modify offsets for the current discontinuous entity
    ner_lists[current_entitiy_idx] = [[ner_lists[current_entitiy_idx][0][0],
                                       ner_lists[current_entitiy_idx][0][1] + len_2nd_fragment],
                                      ner_lists[current_entitiy_idx][-1]]

    for idx in range(len(ner_lists)):
        if idx != current_entitiy_idx:
            for k in range(len(ner_lists[idx]) - 1):
                if (ner_lists[idx][k][0] > last_token_pos_1st_fragment
                        and ner_lists[idx][k][1] > last_token_pos_1st_fragment):
                    ner_lists[idx][k][0] += len_2nd_fragment
                    ner_lists[idx][k][1] += len_2nd_fragment


##############################################################################
########################## Rule 2 ############################################
##############################################################################

# Move the 1st fragment to the left of the 2nd fragment
def modify_tokens_rule_2(tokens, ner_lists, current_entitiy_idx):
    # All tokens before (excluding) the 1st token of the 2nd fragment
    tokens_before_2nd_fragment = tokens[:ner_lists[current_entitiy_idx][1][0]]
    # All tokens of the 1st fragment
    tokens_1st_fragment = tokens[ner_lists[current_entitiy_idx][0][0]:ner_lists[current_entitiy_idx][0][1] + 1]
    # All tokens after (including) the 1st token of the 2nd fragment
    tokens_after_2nd_fragment = tokens[ner_lists[current_entitiy_idx][1][0]:]
    new_tokens = tokens_before_2nd_fragment + tokens_1st_fragment + tokens_after_2nd_fragment
    return new_tokens


def modify_offsets_rule_2(ner_lists, current_entitiy_idx):
    # Number of tokens of the 1st fragment
    len_1st_fragment = ner_lists[current_entitiy_idx][0][1] - ner_lists[current_entitiy_idx][0][0] + 1
    first_token_pos_2nd_fragment = ner_lists[current_entitiy_idx][1][0]
    ner_lists[current_entitiy_idx] = [[ner_lists[current_entitiy_idx][1][0],
                                       ner_lists[current_entitiy_idx][1][1] + len_1st_fragment],
                                      ner_lists[current_entitiy_idx][-1]]
    for idx in range(len(ner_lists)):
        if idx != current_entitiy_idx:
            for k in range(len(ner_lists[idx]) - 1):
                if (ner_lists[idx][k][0] >= first_token_pos_2nd_fragment
                        and ner_lists[idx][k][1] >= first_token_pos_2nd_fragment):
                    ner_lists[idx][k][0] += len_1st_fragment
                    ner_lists[idx][k][1] += len_1st_fragment



##############################################################################
############## Modify sentences with 2-fragment discontinuous entities########
##############################################################################

for doc in range(len(train)):
    tokens = train[doc]['sentences_modified']
    # Changing tokens will NOT change train[doc]['sentences_modified'] for *operations* below
    ner_lists = train[doc]['ner_modified']
    # Changing ner_lists WILL change train[doc]['ner_modified'] for *operations* below
    for current_entitiy_idx in range(len(ner_lists)):
        if len(ner_lists[current_entitiy_idx]) == 3:  # 2-fragment discontinuous entities

            interval_1st = ner_lists[current_entitiy_idx][0]
            interval_2nd = ner_lists[current_entitiy_idx][1]

            switch = 0
            for idx in range(len(ner_lists)):
                if idx != current_entitiy_idx:
                    if len(ner_lists[idx]) == 2:  # Continuous entities
                        interval_cont = ner_lists[idx][0]
                        if check_overlap(interval_2nd, interval_cont):
                            switch = 1
                            break
                        elif check_overlap(interval_1st, interval_cont):
                            switch = 3
                            break

                    elif len(ner_lists[idx]) == 3:  # 2-fragment discontinuous entities
                        interval_discont_1st = ner_lists[idx][0]
                        interval_discont_2nd = ner_lists[idx][1]
                        if (check_overlap(interval_2nd, interval_discont_2nd) is True
                                and check_overlap(interval_1st, interval_discont_1st) is False):
                            switch = 2
                            break
                        elif (check_overlap(interval_1st, interval_discont_1st) is True
                              and check_overlap(interval_2nd, interval_discont_2nd) is False):
                            switch = 4
                            break
                        elif (check_overlap(interval_2nd, interval_discont_2nd) is True
                              and check_overlap(interval_1st, interval_discont_1st) is True
                              and interval_1st[1] > interval_discont_1st[1]
                              and interval_2nd[0] > interval_discont_1st[0]):
                            switch = 5
                            break

            if switch == 1 or switch == 2 or switch == 5:
                tokens = modify_tokens_rule_1(tokens, ner_lists, current_entitiy_idx)
                train[doc]['sentences_modified'] = tokens
                modify_offsets_rule_1(ner_lists, current_entitiy_idx)


            elif switch == 0 or switch == 3 or switch == 4:
                tokens = modify_tokens_rule_2(tokens, ner_lists, current_entitiy_idx)
                train[doc]['sentences_modified'] = tokens
                modify_offsets_rule_2(ner_lists, current_entitiy_idx)


# Sanity check that all entities are equal
for doc in range(len(train)):

    l_modified = []
    for i in range(len(train[doc]['ner_modified'])):
        l_modified.append(train[doc]['sentences_modified'][
                          train[doc]['ner_modified'][i][0][0]:train[doc]['ner_modified'][i][0][1] + 1])

    l_original = []
    for i in range(len(train[doc]['ner'])):
        if len(train[doc]['ner'][i]) == 2:
            l_original.append(train[doc]['sentences'][train[doc]['ner'][i][0][0]:train[doc]['ner'][i][0][1] + 1])
        elif len(train[doc]['ner'][i]) == 3:
            l_original.append(
                train[doc]['sentences'][train[doc]['ner'][i][0][0]:train[doc]['ner'][i][0][1] + 1] + train[doc][
                                                                                                         'sentences'][
                                                                                                     train[doc]['ner'][
                                                                                                         i][1][0]:
                                                                                                     train[doc]['ner'][
                                                                                                         i][1][1] + 1])
        else:
            pass

    if l_modified != l_original:
        print(f'The doc_{doc} needs more examination.')


def create_dict_deleting_discont_morethan2(ner_lists_raw, ner_lists):
    # All indices of discontinuous entities with more than two fragments
    ner_lists_raw_removed = []
    for idx in range(len(ner_lists_raw)):
        if len(ner_lists_raw[idx]) > 3:
            ner_lists_raw_removed.append(idx)

    # Create a dictionary to store the names and associated lists
    dict_ner_lists_raw = {f"T{i + 1}": ner_lists_raw[i] for i in range(len(ner_lists_raw))}

    # Remove the elements in ner_lists_raw_removed from the dictionary
    for idx in ner_lists_raw_removed:
        del dict_ner_lists_raw[f"T{idx + 1}"]

    # Convert the dictionary items to a list
    items = list(dict_ner_lists_raw.items())

    # Create a new dictionary after deleting discontinuous entities with more than two fragments
    new_dict_ner_lists = {}
    for k in range(len(ner_lists)):
        new_dict_ner_lists[items[k][0]] = ner_lists[k]

    return new_dict_ner_lists

###########################################################################
##### Cache entities with double labels needed to remove ##################
###########################################################################

res = []

for doc in range(len(train_re)):

    temp = {}
    doc_name = train_re[doc]['doc']
    temp['doc'] = doc_name
    temp['remove'] = []

    # dictionary to store offsets and their keys
    offset_keys = {}

    # iterate over the list
    for item in train_re[doc]['entities']:
        for k, v in item.items():

            if len(v) == 2:
                offset = tuple(v[0])  # use tuple to make the list hashable
                # store the key instead of the type
                if offset not in offset_keys:
                    offset_keys[offset] = [k]
                else:
                    offset_keys[offset].append(k)

    # find the offsets with more than one key
    for offset, keys in offset_keys.items():
        if len(keys) > 1:
            a = 0
            b = 0
            for l in train_re[doc]['relations']:

                for k, v in l.items():
                    if keys[0] in v:
                        a += 1
                    elif keys[1] in v:
                        b += 1

            l = []
            l.append([keys[0], a])
            l.append([keys[1], b])
            temp['remove'].append(l)
    res.append(temp)

cache = res

# Check whether all entities are continuous or not
for doc in range(len(train)):
    for i in range(len(train[doc]['ner_modified'])):
        if len(train[doc]['ner_modified'][i]) != 2:
            print('There are still discontinuous entities.')

########################################################################
################## Remove entities with double labels ##################
########################################################################

def create_dict_deleting_discont_morethan2(ner_lists_raw, ner_lists):
    # All indices of discontinuous entities with more than two fragments
    ner_lists_raw_removed = []
    for idx in range(len(ner_lists_raw)):
        if len(ner_lists_raw[idx]) > 3:
            ner_lists_raw_removed.append(idx)

    # Create a dictionary to store the names and associated lists
    dict_ner_lists_raw = {f"T{i + 1}": ner_lists_raw[i] for i in range(len(ner_lists_raw))}

    # Remove the elements in ner_lists_raw_removed from the dictionary
    for idx in ner_lists_raw_removed:
        del dict_ner_lists_raw[f"T{idx + 1}"]

    # Convert the dictionary items to a list
    items = list(dict_ner_lists_raw.items())

    # Create a new dictionary after deleting discontinuous entities with more than two fragments
    new_dict_ner_lists = {}
    for k in range(len(ner_lists)):
        new_dict_ner_lists[items[k][0]] = ner_lists[k]

    return new_dict_ner_lists


for doc in range(len(train)):

    train[doc]['relations'] = []
    ner_lists_raw = train[doc]['ner']
    ner_lists = train[doc]['ner_modified']
    new_dict_ner_lists = create_dict_deleting_discont_morethan2(ner_lists_raw, ner_lists)

    # Add relations
    if 'relations' in train_re[doc]:
        for j in range(len(train_re[doc]['relations'])):
            for key, value in train_re[doc]['relations'][j].items():
                if value[0] in new_dict_ner_lists and value[1] in new_dict_ner_lists:
                    rel_temp = []
                    rel_temp += (new_dict_ner_lists[value[0]][0] + new_dict_ner_lists[value[1]][0])
                    rel_temp.append(key)
            train[doc]['relations'].append(rel_temp)

    if cache[doc]['remove'] != []:  # Need to remove a few entities and relations
        for i in range(len(cache[doc]['remove'])):
            a_num = cache[doc]['remove'][i][0][1]
            a_name = cache[doc]['remove'][i][0][0]
            b_num = cache[doc]['remove'][i][1][1]
            b_name = cache[doc]['remove'][i][1][0]

            if a_num == 0 and b_num == 0:
                chosen_element = random.choice([0, 1])
                remove_name = cache[doc]['remove'][i][chosen_element][0]
                remove_offsets = new_dict_ner_lists[remove_name]
                train[doc]['ner_modified'].remove(remove_offsets)


            elif (a_num == 0 and b_num >= 1) or (b_num == 0 and a_num >= 1):
                if a_num == 0:
                    remove_offsets = new_dict_ner_lists[a_name]
                    train[doc]['ner_modified'].remove(remove_offsets)
                elif b_num == 0:
                    remove_offsets = new_dict_ner_lists[b_name]
                    train[doc]['ner_modified'].remove(remove_offsets)


            elif a_num == 1 and b_num == 1:
                chosen_element = random.choice([0, 1])
                remove_name = cache[doc]['remove'][i][chosen_element][0]
                remove_offsets = new_dict_ner_lists[remove_name]
                train[doc]['ner_modified'].remove(remove_offsets)

                result = []
                for dictionary in train_re[doc]['relations']:
                    for key, values in dictionary.items():
                        if remove_name in values:
                            result.append([values[0], values[1], key])

                for item in result:
                    rel_temp = []
                    rel_temp += (new_dict_ner_lists[item[0]][0] + new_dict_ner_lists[item[1]][0])
                    rel_temp.append(item[2])
                    train[doc]['relations'].remove(rel_temp)

            else:  # (a_num = 1, b_num = N) or (a_num = N, b_num = 1)
                if a_num == 1:
                    remove_offsets = new_dict_ner_lists[a_name]
                    train[doc]['ner_modified'].remove(remove_offsets)

                    result = []
                    for dictionary in train_re[doc]['relations']:
                        for key, values in dictionary.items():
                            if a_name in values:
                                result.append([values[0], values[1], key])

                    for item in result:
                        rel_temp = []
                        rel_temp += (new_dict_ner_lists[item[0]][0] + new_dict_ner_lists[item[1]][0])
                        rel_temp.append(item[2])

                        train[doc]['relations'].remove(rel_temp)

                elif b_num == 1:
                    remove_offsets = new_dict_ner_lists[b_name]
                    train[doc]['ner_modified'].remove(remove_offsets)

                    result = []
                    for dictionary in train_re[doc]['relations']:
                        for key, values in dictionary.items():
                            if b_name in values:
                                result.append([values[0], values[1], key])

                    for item in result:
                        rel_temp = []
                        rel_temp += (new_dict_ner_lists[item[0]][0] + new_dict_ner_lists[item[1]][0])
                        rel_temp.append(item[2])

                        train[doc]['relations'].remove(rel_temp)



# Addjust the format to suit for PURE

for i in range(len(train)):
    train[i].pop('sentences')
    train[i].pop('ner')
    train[i]['sentences'] = train[i].pop('sentences_modified')
    train[i]['ner'] = train[i].pop('ner_modified')
    train[i]['sentences'] = [train[i]['sentences']]
    train[i]['ner'] = [train[i]['ner']]
    train[i]['relations'] = [train[i]['relations']]


with open(output_file_path, "w") as f_out:
    for line in train:
        f_out.write(json.dumps(line))
        f_out.write('\n')