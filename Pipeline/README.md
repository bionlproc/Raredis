```train.json``` and ```dev.json``` have only gold entities.

```train_re.json``` and ```dev_re.json``` have gold entities and relations.

```test.json``` has predicted entities from SODNER: continuous entities (with predicted entity types) + discontinuous entities (with "UNKNOWN" entity type). It does not contain predicted relations.

```preprocess_train.py``` preprocesses ```train.json``` and ```train_re.json``` from ```raw_data/train``` into ```preprocessed_data```.

```preprocess_dev.py``` preprocesses ```dev.json``` and ```dev_re.json``` from ```raw_data/dev``` into ```preprocessed_data```.

```preprocess_test.py``` preprocesses ```test.json``` from ```raw_data/test``` into ```preprocessed_data```.
