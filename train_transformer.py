import os
import time
import torch
import torch.optim as optim
import numpy as np

from src import data_ops
from src import loss_functions
from src.checkpoints import load_checkpoint, save_checkpoint
from src.transformer import EncoderTransformer

# load tokenizer
from tokenizers import Tokenizer

import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


### Define Parameters
## Tokenizer
# tokenizer_filepath = f".tokenizers/tok_SL128_V15000.json"
tokenizer_filepath = f".tokenizers/tok_SL128_V15000.json"
tokenizer = Tokenizer.from_file(tokenizer_filepath)
vocab_size = tokenizer.get_vocab_size()
sequence_length = tokenizer.truncation["max_length"]
padding_idx = tokenizer.encode("[PAD]").ids[0]
mask_idx = tokenizer.encode("[MASK]").ids[0]

## Model
d_model = 512
d_embedding = d_model
d_ff = d_model * 4
n_layers = 8
dropout = 0.1

d_k = 64  # k and v dims per head
d_v = d_k
n_heads = d_model / d_k
if n_heads != int(n_heads):
    raise ValueError("d_model must be divisible by d_k")
n_heads = int(n_heads)

## Training
# data_dir = "D://data/embedded_text/wikipedia_vocab64_seqlen15k"
data_dir = ".data\\tokenized_test_128"
batch_size = 320
n_epochs = 20
lr = 2e-4
weight_decay = 0.0

## Masking
mask_probability = 0.15
proportion_mask_token = 0.8
proportion_random_token = 0.1
# proportion_mask_token = 0.0
# proportion_random_token = 0.0

# Checkpointing
FLAG_LOAD_CHECKPOINT = True
checkpoint_dir = ".checkpoints"
checkpoint_epoch_dir = os.path.join(checkpoint_dir, "epoch")
checkpoint_every = 50
max_checkpoints = 5
checkpoint_relpath = "epoch1_file0.pt"
# checkpoint_relpath = "epoch/epoch6_file0.pt"

transformer = EncoderTransformer(
    vocab_size,
    sequence_length,
    n_layers,
    d_embedding,
    d_model,
    d_k,
    d_v,
    n_heads,
    d_ff,
    padding_idx,
    dropout,
).to(device)

if FLAG_LOAD_CHECKPOINT:
    optimizer = optim.Adam(transformer.parameters())
    data_loader = data_ops.DataLoader(data_dir, find_files=False, device=device)
    start_epoch = load_checkpoint(
        os.path.join(checkpoint_dir, checkpoint_relpath),
        transformer,
        optimizer,
        data_loader,
    )
    file_idx = data_loader.file_idx
    initial_file_idx = file_idx
    last_checkpoint_idx = file_idx
else:
    data_ops.create_directory(checkpoint_dir, reset=True)
    data_ops.create_directory(checkpoint_epoch_dir, reset=True)

    optimizer = optim.Adam(transformer.parameters(), lr=lr, weight_decay=weight_decay)

    start_epoch = 0
    file_idx = 0
    initial_file_idx = 0

    # Keep only the relative path from the data directory
    data_loader = data_ops.DataLoader(data_dir, batch_size, device=device)

    last_checkpoint_idx = 0

wandb.init(
    project="transformer-encoder",
    config={
        "dataset": data_dir,
        "tokenizer": tokenizer_filepath,
        "vocab_size": vocab_size,
        "sequence_length": sequence_length,
        "model_dimension": d_model,
        "embedding_dimension": d_embedding,
        "feedforward_dimension": d_ff,
        "num_layers": n_layers,
        "dropout_rate": dropout,
        "k_dimension": d_k,
        "v_dimension": d_v,
        "num_heads": n_heads,
        "batch_size": batch_size,
        "num_epochs": n_epochs,
        "learning_rate": lr,
        "mask_probability": mask_probability,
        "proportion_mask_token": proportion_mask_token,
        "proportion_random_token": proportion_random_token,
    },
)

print("Training transformer model:")
print(transformer)
print(
    f"Number of parameters: {sum(p.numel() for p in transformer.parameters() if p.requires_grad)}\n"
)


def print_progress(total_start, checkpoint_start, total_n_files, checkpoint_n_files):
    if total_n_files == 0:
        return
    total_elapsed = time.time() - total_start
    checkpoint_elapsed = time.time() - checkpoint_start
    total_avg_time = total_elapsed / total_n_files
    checkpoint_avg_time = (
        checkpoint_elapsed / checkpoint_n_files if checkpoint_n_files > 0 else 0
    )

    print(
        f"Total: {data_ops.seconds_to_ms(total_elapsed)} ({total_avg_time:.2f}s/file) | Checkpoint: {data_ops.seconds_to_ms(checkpoint_elapsed)} ({checkpoint_avg_time:.2f}s/file)"
    )


for epoch in range(start_epoch, n_epochs):
    print(f"Epoch {epoch}/{n_epochs-1}")
    start = time.time()
    checkpoint_start = time.time()

    for batch, batch_counter, file_idx in data_loader:
        # Checkpoint
        if file_idx % checkpoint_every == 0 and file_idx > last_checkpoint_idx:
            save_checkpoint(
                checkpoint_dir,
                transformer,
                optimizer,
                data_loader,
                epoch,
                file_idx,
                max_checkpoints,
            )
            print_progress(
                start,
                checkpoint_start,
                file_idx - initial_file_idx,
                file_idx - last_checkpoint_idx,
            )
            checkpoint_start = time.time()
            last_checkpoint_idx = file_idx

        optimizer.zero_grad()

        ## Mask tokens
        masked_batch, masked_batch_bool = data_ops.mask_tokens(
            batch,
            mask_id=mask_idx,
            pad_id=padding_idx,
            mask_prob=mask_probability,
            vocab_low_high=(5, vocab_size),
            proportion_mask_token=proportion_mask_token,
            proportion_random_token=proportion_random_token,
        )

        ## Forward pass
        all_token_likelihoods = transformer(masked_batch)  # output: b,s,voc

        ## Calculate loss
        loss = loss_functions.cross_entropy(
            all_token_likelihoods, masked_batch, masked_batch_bool
        )

        ## Optimize
        loss.backward()
        optimizer.step()
        wandb.log(
            {
                "epoch": epoch,
                "batch_number": batch_counter,
                "cross_entropy": loss.item(),
            }
        )

    ## Reset loaded variables
    initial_file_idx = 0
    last_checkpoint_idx = -1  # to ensure the first checkpoint is saved


## Save final model
save_checkpoint(
    checkpoint_epoch_dir,
    transformer,
    optimizer,
    data_loader,
    epoch + 1,
    file_idx=0,
    max_checkpoints=max_checkpoints,
)
print_progress(
    start,
    checkpoint_start,
    file_idx - initial_file_idx,
    file_idx - last_checkpoint_idx,
)
