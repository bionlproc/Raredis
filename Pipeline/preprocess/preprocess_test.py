import json

Path_train = '/Users/XA/Desktop/Raredis/Pipeline/raw_data/test/test.json'
output_file_path = "/Users/XA/Desktop/Raredis/Pipeline/preprocessed_data/test.json"

train = []
for line in open(Path_train, 'r'):
    train.append(json.loads(line))

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


# Addjust the format to suit for PURE
for i in range(len(train)):
    train[i].pop('sentences')
    train[i].pop('ner')
    train[i]['sentences'] = train[i].pop('sentences_modified')
    train[i]['ner'] = train[i].pop('ner_modified')

def flatten_list(nested_list):
    flattened_list = []
    for sublist in nested_list:
        if isinstance(sublist, list):
            flattened_list.extend(sublist)
        else:
            flattened_list.append(sublist)
    return flattened_list

for i in range(len(train)):
    for j in range(len(train[i]['ner'])):
        train[i]['ner'][j] = flatten_list(train[i]['ner'][j])

for i in range(len(train)):
    train[i]['relations'] = [[]]

for i in range(len(train)):
    train[i]['sentences'] = [train[i]['sentences']]
    train[i]['ner'] = [train[i]['ner']]

with open(output_file_path, "w") as f_out:
    for line in train:
        f_out.write(json.dumps(line))
        f_out.write('\n')