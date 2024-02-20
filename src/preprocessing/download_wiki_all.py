"""
Download and preprocess Wikipedia dataset for training language models.
"""

import os
from datasets import load_dataset
from spacy.lang.en import English

output_dir = os.path.join('.data','raw_text')

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("Loading Wikipedia dataset")
dataset = load_dataset("wikipedia", "20220301.en", split="train", trust_remote_code=True)

print("Loaded spacy sentencizer")
nlp = English()
nlp.add_pipe("sentencizer")

# Shard the file into groups, one row per sentence
number_of_shards = 1000
print("Extracting shards...")
for i in range(number_of_shards):
    filename = str(i)
    # print(f'Writing {filename}')
    with open(os.path.join(output_dir, filename), 'w', encoding='utf-8') as f:
        for item in dataset.shard(number_of_shards, i)['text']:
            item = item.replace('\n', ' ')
            # doc = nlp(item)
            # for sent in doc.sents:
            f.write("%s\n" % item)
    print(f'{i+1}/{number_of_shards} shards processed', end='\r')

    if i == 214:
        break