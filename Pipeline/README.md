Raw data contains train / dev / test datasets.

For the train dataset, there are 2 slightly different datasets. train.json only has gold entities. train_re.json has gold entities + gold relations.
For the dev dataset, there are 2 slightly different datasets. dev.json only has gold entities. dev_re.json has gold entities + gold relations.
For the test dataset, there is one dataset test.json from SODNER. It has entity predictions: continuous entities (with predicted entity types) + discontinuous entities (with "UNKNOWN" entity type). It does not contain predicted relations.