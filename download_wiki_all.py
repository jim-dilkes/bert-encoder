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

output_dir = './.data/data_sent'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("Loading Wikipedia dataset")
dataset = load_dataset("wikipedia", "20220301.en", split="train", trust_remote_code=True)
    
nlp = English()
nlp.add_pipe("sentencizer")

# Shard the file into groups, one row per sentence
number_of_shards = 1000
print("Extracting shards...")
for i in range(number_of_shards):
    filename = f'wikipedia_ALL_{str(i)}.txt' 
    # print(f'Writing {filename}')
    with open(os.path.join(output_dir, filename), 'w', encoding='utf-8') as f:
        for item in dataset.shard(number_of_shards, i)['text']:
            item = item.replace('\n', ' ')
            doc = nlp(item)
            for sent in doc.sents:
                f.write("%s\n" % sent.text)
    print(f'{i+1}/{number_of_shards} shards processed', end='\r')