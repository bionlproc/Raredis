## Run scripts

Run zero-shot E2ERE on the test set.

```
python main.py run
--model {'gpt-4-1106-preview | o1'}
--dataset_name Raredis
--split test
--openai_key {Your API key}
--max_examples None
--template RaredisTemplate_json
--save_dir {Directory of output files}
--max_tokens {8192}
--temperatures {0.1}
```
