import torch
import pandas as pd
import json

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

fine_tuned_model = "/home/ubuntu/FlanT5/Raredis/output/Flant5_xl/nocopy_relis/checkpoint-400"
model = AutoModelForSeq2SeqLM.from_pretrained(fine_tuned_model)
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl")
model = model.to(device)
print("Loading test file")
with open('/home/ubuntu/FlanT5/Raredis/data/no_copy_relis/test.json', 'r') as file:
    test_data = json.load(file)

# Separate source and target texts
test_source_texts = [item['source'] for item in test_data]
test_target_texts = [item['target'] for item in test_data]

df = pd.DataFrame(columns=['source', 'gold', 'prediction'])
for i,x in enumerate(zip(test_source_texts,test_target_texts)):
    inputs = tokenizer(x[0], max_length=1024, padding="max_length", truncation=True, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=1024)
    prediction = tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    # df = df.append({'source': x[0], "gold": x[1], 'prediction': prediction}, ignore_index=True)
    df = pd.concat([df, pd.DataFrame([{'source': x[0], "gold": x[1], 'prediction': prediction}])], ignore_index=True)
    print(i, prediction, "\n")

df.to_csv("/home/ubuntu/FlanT5/Raredis/output/Flant5_xl/nocopy_relis/prediction/predictions.csv", index=False)