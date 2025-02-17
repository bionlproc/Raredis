# Error Analysis
We note that many RE errors appear to arise from NER errors. This can lead to a snowball effect
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

Discontinuous entities are particularly tricky and have led to several er-
rors, even if the prediction is not incorrect, because the model was unable to split an entity conjunction into
constituent entities. Consider the snippet: *“affected infants may exhibit abnormally long, thin fingers and toes
and/or deformed (dysplastic) or absent nails at birth.”* Instead of generating relations with the two gold entities
“abnormally long, thin fingers” and “abnormally long, thin toes”, the model simply created one relation with
“long, thin fingers and toes.”

## BioMedLM generations not in the input

In several cases we noticed spans that were not in the input but were
nevertheless closely linked with the gold entity span’s meaning. For example, for the gold span “muscle twitch-
ing”, BioMedLM predicted “muscle weakness”. It also tried to form meaningful noun phrases that capture the
meaning of longer gold spans. For instance, for the gold span “ability to speak impaired”, it predicted “difficulty
in speaking”. For the gold span, “progressive weakness of the muscles of the legs” it outputs “paralysis of the
legs”. All these lead to both FPs and FNs, unfortunately.

## Errors due to potential annotation issues

In document-level RE settings, it is not uncommon for annotators
to miss certain relations. But when these are predicted by a model, they would be considered FPs. Consider the
context: *“The symptoms of infectious arthritis depend upon which agent has caused the infection but symptoms
often include fever, chills, general weakness, and headaches.”* Our model predicted that “infectious arthritis”
produces “fever”. However, the gold predictions for this did not have this and instead had the relation “the
infection” (anaphor) produces “fever”. While the gold relation is correct, we believe what our model extracted
is more meaningful. However, since we missed the anaphor-involved relation, it led to an FN and an FP.
