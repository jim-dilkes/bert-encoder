from ..src import data_ops
import os
import torch

from tokenizers import Tokenizer


shard_size = 10000
parent_dir = os.path.join(".", ".data", "data_sent")
output_dir = os.path.join(".", ".data", "data_tokenized")

# Get the file names
file_names = os.listdir(parent_dir)
# Create the output directory (delete if it already exists)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
# data_ops.create_directory(output_dir)


# Load the tokenizer
tokenizer = Tokenizer.from_file("tokenizer_V2.json")


def tokenize_sentences(batch: list[str]) -> torch.Tensor:
    tokenized = tokenizer.encode_batch(batch)
    tokenized = torch.tensor([x.ids for x in tokenized])
    return tokenized


for i, file_name in enumerate(file_names):
    os.makedirs(os.path.join(output_dir, str(i)), exist_ok=True)


    with open(os.path.join(parent_dir, file_name), "r", encoding="utf-8") as f:
        shard = []
        j=0
        for line in f.readlines():
            shard.append(line)

            # Once the output file reaches the desired size, tokenize the sentences and save them to a file
            if len(shard) == shard_size:
                print(f"Tokenizing {i+1}/{len(file_names)} files, done {j*shard_size}", end="\r")

                shard_tokenized = tokenize_sentences(shard)
                torch.save(shard_tokenized, os.path.join(output_dir, str(i), f"{j}.pt"))

                shard = []
                j+=1

        # Tokenize the remaining sentences                
        if len(shard) > 0:
            shard_tokenized = tokenize_sentences(shard)
            torch.save(shard_tokenized, os.path.join(output_dir, str(i), f"{j}.pt"))

    
    # Delete the original file
    os.remove(os.path.join(parent_dir, file_name))    

    print(f"Tokenized {i+1}/{len(file_names)} files", end="\r")