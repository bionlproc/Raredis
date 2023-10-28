import json

# Directory for the SODNER prediction file
prediction_dir = "/Users/shashankgupta/Documents/Raredis/SODNER/trained_models/latest_prediction/prediction2.txt"

# Open the text file in read mode
with open(prediction_dir, 'r') as file:
    # Read the file line by line and store each line in a list
    lines = file.readlines()

# List to dump as JSON as final output file
output_list = []

# Looping predictions by the SODNER
for line in lines:
    new_dict = {}
    dictionary = json.loads(line)
    predicted_entities = dictionary["prediction"][0]["ner"]
    entities_list = []

    predicted_rels = dictionary["prediction"][0]["relation"]
    if len(predicted_rels) > 0:
        rels_list = []
        for idx, rels in enumerate(predicted_rels):
            if rels[-1] == "Combined":
                indexes = rels[:4]
                sorted_indexes = sorted(indexes)
                if sorted_indexes not in rels_list:
                    rels_list.append(sorted_indexes)

        for discont in rels_list:
            first_two = discont[:2]
            last_two = discont[2:]

            indexes_to_delete = []
            for idx_1, x in enumerate(predicted_entities):
                span = x[:2]
                if span == first_two:
                    indexes_to_delete.append(idx_1)
                elif span == last_two:
                    indexes_to_delete.append(idx_1)

            for index in sorted(indexes_to_delete, reverse=True):
                del predicted_entities[index]

        for disc in rels_list:
            rels_dict = dict()
            element = [str(disc[0]) + "," + str(disc[1]), str(disc[2]) + "," + str(disc[3])]
            rels_dict["type"] = "UNKNOWN"
            rels_dict["span"] = element
            entities_list.append(rels_dict)

    if len(predicted_entities) > 0:
        for flat_ent in predicted_entities:
            ent_dict = dict()
            ent_dict["type"] = flat_ent[-1]
            start = str(flat_ent[0])
            end = str(flat_ent[1])
            ent_dict["span"] = [start + "," + end]
            entities_list.append(ent_dict)

    new_dict["doc"] = dictionary["doc_key"]
    new_dict["tokens"] = dictionary["sentences"][0]
    new_dict["text"] = ' '.join(dictionary["sentences"][0])
    new_dict["start"] = -1
    new_dict["end"] = -1

    sorted_entities_list = sorted(entities_list, key=lambda x: int(x["span"][0].split(',')[0]))
    new_dict["entities"] = sorted_entities_list
    output_list.append(new_dict)

# # Open a file in write mode
# with open('/Users/shashankgupta/Documents/Raredis/SODNER/trained_models/latest_prediction/final_prediction.json', 'w') as json_file:
#     # Iterate over the dictionaries
#     for dictionary in output_list:
#         # Dump each dictionary into the file
#         json.dump(dictionary, json_file)
#         json_file.write('\n')  # Add a newline after each dictionary
