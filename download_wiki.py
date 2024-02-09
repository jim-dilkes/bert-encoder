"""
Download and preprocess Wikipedia dataset for training language models.
Preprocessing includes:
- Selecting a subset of the dataset
- Shuffling the dataset
- Splitting the dataset into smaller files
- Splitting the dataset into sentences using spacy
"""

import os
from datasets import load_dataset
from spacy.lang.en import English
import random

output_dir = 'data_sent'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

n_select = int(1e6)
n_sample = int(1e5)
n_shard = int(1e4)


print("Loading Wikipedia dataset")
dataset = load_dataset("wikipedia", "20220301.en", split="train", trust_remote_code=True)

print(f"Shuffling and selecting {n_select} items")
dataset.shuffle(seed=1)
dataset = dataset.select(range(n_select))
dataset = dataset['text']

print(f"Sample {n_sample} items")
random.seed(1)
dataset = random.sample(dataset, n_sample)



nlp = English()
nlp.add_pipe("sentencizer")

# Shard the file into groups, one row per sentence
i=0
while dataset:
    filename = f'wikipedia_{int(n_sample/1000)}K_{str(i)}.txt'
    print(f'Writing {filename}')
    with open(os.path.join(output_dir, filename), 'w', encoding='utf-8') as f:
        for item in dataset[:n_shard]:
            item = item.replace('\n', ' ')
            doc = nlp(item)
            for sent in doc.sents:
                f.write("%s\n" % sent.text)
    dataset = dataset[n_shard:]
    i+=1