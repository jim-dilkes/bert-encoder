"""
Download and preprocess a dataset for training language models.
"""

import os
import argparse
from .. import data_ops

from datasets import load_dataset
from spacy.lang.en import English

from tokenizers import Tokenizer
import torch
import random

from concurrent.futures import ProcessPoolExecutor


def tokenize_sentences(tokenizer: Tokenizer, batch: list[str]) -> torch.Tensor:
    tokenized = tokenizer.encode_batch(batch)
    tokenized = torch.tensor([x.ids for x in tokenized])
    return tokenized

def process_text(text, nlp):
    # Process text with SpaCy nlp
    return [sent.text for sent in nlp(text).sents]

def main(dataset_name:str, dataset_version:str, split:str, output_dir:str, reset_dir:bool, 
         tokenizer_filepath:str, sentencize:bool, n_shards:int, file_size:int,
            shuffle_dataset:bool, shuffle_shards:bool, seed:int):
    
    data_ops.create_directory(output_dir, reset_dir)

    print(f"Loading dataset: {dataset_name}, {dataset_version}, {split}")
    dataset = load_dataset(dataset_name, dataset_version, split=split, trust_remote_code=True)

    if shuffle_dataset:
        print("Shuffling dataset...")
        dataset = dataset.shuffle(seed=seed)

    # Load the spacy sentencizer   
    if sentencize:
        print("Loading spaCy  sentencizer...")
        nlp = English()
        nlp.add_pipe("sentencizer")

    # Load the tokenizer
    print("Loading tokenizer...")
    tokenizer = Tokenizer.from_file(tokenizer_filepath)

    # Iterate through the shards
    print("Extracting shards...")
    for i in range(n_shards):
        # Load all the sentences in the shard
        all_sents = []
        print(f"Loading shard {i+1}/{n_shards}...")
        all_sents = dataset.shard(n_shards, i)['text']
        if sentencize:
            all_sents = list(sent.text for doc in nlp.pipe(all_sents, disable=["parser", "tagger", "ner"]) for sent in doc.sents)
        

        # Shuffle the shard
        if shuffle_shards:
            print(f"Shuffling shard {i+1}/{n_shards}...")
            all_sents = random.sample(all_sents, len(all_sents))


        # One dir per shard
        data_ops.create_directory(os.path.join(output_dir, str(i)), reset_dir)
        next_file = []
        j=0 # File index within the shard
        # Iterate through the loaded shard and save the sentences to a file
        while len(all_sents) > 0:
            next_file = all_sents[:file_size]
            all_sents = all_sents[file_size:]

            print(f"Tokenizing shard {i+1}/{n_shards}, done {round((j*file_size)/1000.0)}K sentences", end="\r")

            shard_tokenized = tokenize_sentences(tokenizer, next_file)
            torch.save(shard_tokenized, os.path.join(output_dir, str(i), f"{j}.pt"))

            next_file = []
            j+=1

        print(f'\n{i+1}/{n_shards} shards processed')


# --output_dir .data/tokenized --tokenizer tokenizer_V2.json --shuffle_dataset --shuffle_shards
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download and preprocess (sentencise, tokenize) the given HuggingFace Dataset for training language models.')
    parser.add_argument('--output_dir', type=str, help='The output directory for the transformed dataset')
    parser.add_argument('--reset_dir', action='store_true', help='Whether to reset the output directory')
    parser.add_argument('--dataset_name', type=str, help='The name of the dataset to download', default='wikipedia')
    parser.add_argument('--dataset_version', type=str, help='The version of the dataset to download', default='20220301.en')
    parser.add_argument('--dataset_split', type=str, help='The name of the split to download', default='train')
    parser.add_argument('--tokenizer', type=str, help='The filepath of the tokenizer to use', default='tokenizer.json')
    parser.add_argument('--without_sentencize', action='store_false', help='Whether to sentencize the text')
    parser.add_argument('--n_shards', type=int, help='The number of shards to split the dataset into', default=1000)
    parser.add_argument('--file_size', type=int, help='The number of sentences per file', default=10000)
    parser.add_argument('--shuffle_dataset', action='store_true', help='Whether to shuffle the dataset before splitting into shards')
    parser.add_argument('--shuffle_shards', action='store_true', help='Whether to shuffle the shards before saving to file')
    parser.add_argument('--seed', type=int, help='The random seed to use for shuffling', default=0)

    args = parser.parse_args()

    main(args.dataset_name, args.dataset_version, args.dataset_split, args.output_dir, args.reset_dir,
            args.tokenizer, args.without_sentencize, args.n_shards, args.file_size,
            args.shuffle_dataset, args.shuffle_shards, args.seed)