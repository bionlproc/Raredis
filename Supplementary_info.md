# Predicate-specific Analysis
We also wanted to examine scores per relation type in our models to see if there are any predicates for which we are underperforming more than expected. From the following table, we notice that recall is less than 5% for 
"increases_risk_of" relation type. This is quite awful but not surprising given the prevalence of such relations is very small in the dataset (from Table 1 in the paper). But what is very unusual is the F1 of the "produces" relation being less than 50, when it constitutes over 60% of all relations in the dataset (from Table 1 in the paper). Upon deeper investigation, we found that generally longer object entities lead to NER errors. We checked this more concretely by examining the errors (for 'produces') and found out that we missed 43% of the object spans for the best-performing pipeline method. Thus, a large portion of performance loss is simply due to the model not being able to predict the object entity span correctly; especially for long object entities, even missing a single token can lead to RE errors.


| Relation type       | P (SODNER+PURE) | R (SODNER+PURE) | F (SODNER+PURE) | P (Seq2Rel) | R (Seq2Rel) | F (Seq2Rel) | P (BioMedLM) | R (BioMedLM) | F (BioMedLM) | P (Flan-T5-large) | R (Flan-T5-large) | F (Flan-T5-large) |
|---------------------|----------------|----------------|----------------|-------------|-------------|-------------|-------------|-------------|-------------|-------------------|-------------------|-------------------|
| anaphora           | 70.40          | 69.84          | **70.11**      | 64.60       | 58.00       | 61.08       | 61.26       | 53.96       | 57.38       | 62.99            | 63.49            | 63.24            |
| is_a              | 62.67          | 55.29          | **58.75**      | 58.67       | 51.76       | 55.00       | 52.77       | 44.70       | 48.40       | 61.84            | 55.29            | 58.38            |
| is_acron         | 70.37          | 57.58          | **63.33**      | 50.00       | 42.00       | 45.65       | 55.17       | 48.48       | 51.61       | 59.25            | 48.48            | 53.33            |
| produces         | 50.21          | 45.09          | **47.51**      | 47.48       | 41.13       | 44.00       | 37.20       | 32.82       | 34.87       | 43.05            | 43.45            | 43.24            |
| is_synon         | 75.00          | 18.75          | **30.00**      | 100.00      | 12.50       | 22.23       | 0.00        | 0.00        | 0.00        | 0.00             | 0.00             | 0.00             |
| increases_risk_of | 50.00          | 4.55           | 8.33           | 11.80       | 9.52        | **10.52**   | 0.00        | 0.00        | 0.00        | 0.00             | 0.00             | 0.00             |



# Error Analysis
Before we proceed, we note that many RE errors appear to arise
from NER errors. This can lead to a snowball effect
of errors in the RE phase. Consider a single entity participating in n gold relations. If it is predicted incorrectly as a
partial match, it may potentially lead to 2n relation errors because it can give rise to n false positives (FPs) (because
the relation is predicted with the wrong span) and n false negatives (FNs) (because the gold relation with the right span
is missed). Thus, even a small proportion of NER errors can lead to a high loss in RE performance. In this section, we
discuss a few error categories that we observed commonly across models.

## Partial matches

When multi-word entities are involved, the relation error is often due to the model predicting a
partial match (a substring or superstring of a gold span) and this was frequent in our effort. Consider the snippet
“Kienbock disease changes may produce pain...The range of motion may become restricted”. Here Kienbock
disease is the subject of a produces relation with the gold object span: “the range of motion may become
restricted”. However, the Seq2Rel model predicted “range of motion restricted” as the object span, leading to
both an FP and FN. But common sense tells us that the model prediction is also correct (and potentially even
better) because it removed the unnecessary “may become” substring. In a different example, when the relation
involved the gold span “neurological disorder,” the model predicted a superstring “progressive neurological disorder” 
from the full context: “Subacute sclerosing panencephalitis (SSPE) is a progressive neurological
disorder.”

## Entity type mismatch

Because our evaluation is strict, predicting the entity spans and relation type correctly,
but missing a single entity type can invalidate the whole relation leading to both an FP and an FN. The models
are often confused between closely related entity types. **Rare disease** and **skin rare disease** were often confused
along with the pair **sign** and **symptom**.

## Issues with discontinuous entities

Tricky discontinuous entities  have led to several errors, even if the prediction is not incorrect, because the model was unable to split an entity conjunction into constituent entities. Consider the snippet: *"affected infants may exhibit abnormally long, thin fingers and toes and/or deformed (dysplastic) or absent nails at birth."* Instead of generating relations with the two gold entities "abnormally long, thin fingers" and "abnormally long, thin toes", the model simply created one relation with "long, thin fingers and toes."

## BioMedLM generations not in the input

In several cases we noticed spans that were not in the input but were
nevertheless closely linked with the gold entity span’s meaning. For example, for the gold span “muscle twitching”, 
BioMedLM predicted “muscle weakness”. It also tried to form meaningful noun phrases that capture the
meaning of longer gold spans. For instance, for the gold span “ability to speak impaired”, it predicted “difficulty
in speaking”. For the gold span, “progressive weakness of the muscles of the legs” it outputs “paralysis of the
legs”. All these lead to both FPs and FNs, unfortunately.

## Errors due to potential annotation issues

In document-level RE settings, it is not uncommon for annotators to miss certain relations. But when these are predicted by a model, they would be considered FPs. Consider the context: *"The symptoms of infectious arthritis depend upon which agent has caused the infection but symptoms often include fever, chills, general weakness, and headaches."* Our model predicted that "infectious arthritis" *produces* "fever". However, the gold predictions for this did not have this and instead had the relation "the infection" (anaphor) *produces* "fever". While the gold relation is correct, we believe what our model extracted is more meaningful. However, since we missed the anaphor-involved relation, it led to an FN and an FP. 

# Model Configuration Details
Experiments for the pipeline approach were performed on our inhouse cluster of 32GB GPUs. All experiments for Seq2Rel were performed on Google Colab Pro+ using an Nvidia a100-sxm4-40GB GPU with access to high RAM. In Seq2Rel, we use AllenNLP, an open-source NLP library developed by the Allen Institute for Artificial Intelligence (AI2). Fairseq, a sequence modeling toolkit, is used for training custom models for text generation tasks for BioGPT on Google Colab Pro. We fine-tuned BioMedLM on a single H100 80GB GPU. Next we describe the hyperparameters chosen for each of the models based on validation dataset.

## Pipeline (SODNER+PURE)
We used a batch size of 8, a learning rate of 1e-3, and 100 epochs to train the SODNER model for discontinuous entities with a PubMedBERT-base encoder. For the PURE NER model, we used PubMedBERT-base and trained for 100 epochs, with a learning rate of 1e-4 and a batch size of 8. We also experimented with PubMedBERT-large with the same settings. For the PURE relation model,  we used both PubMedBERT-base and PubMedBERT-large as encoders with a learning rate of 1e-5  and trained for 25 epochs with the training batch size of 8. 

## Seq2Rel
Training was conducted for 150 epochs, with a learning rate of 2e-5 for the PubMedBERT encoders  and 1.21e-4 for the decoder (LSTM) with a batch size of 2 and a beam size of 3 (for the decoder).

## BioMedLM
Despite supervised fine-tuning, it is not uncommon for GPT models to output strings that were not part of the input. We observed that nearly 3%-7% of entities output by BioMedLM did not exactly match ground truth spans. Since we require an exact match for a prediction to be correct, we appended explicit natural language instructions to the input, directing the model to generate tokens from the input text: "From the given abstract, find all the entities and relations among them. Do not generate any token outside the abstract." We used a batch size of 1 with gradient_accumulation_steps of 16, a learning rate of 1e-5, and 30 epochs for BioMedLM.

## T5
With the same output templates used for BioMedLM, we trained T5-3B, Flan-T5-Large (770M), and Flan-T5-XL (3B). For T5-3B, we used  batch size one with gradient_accumulation_steps set to 16, lr = 3e-4, 100 epochs, and generation beam size of 4. For Flan-T5, we used a batch size two with gradient_accumulation_steps set to 16, and the rest of the hyperparameters same as T5-3B. For Flan-T5-XL, we had gradient_accumulation_steps set to 16, batch size one, lr = 3e-4, 100 epochs, and generation beam size of 4 with DeepSpeed for CPU offloading of the parameters.


