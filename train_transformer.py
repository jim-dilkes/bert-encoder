import math
import os
import time
import torch
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

from src import data_ops
from src.transformer import EncoderTransformer

import numpy as np

# load tokenizer
from tokenizers import Tokenizer

tokenizer_filepath = ".tokenizers/tokenizer_64.json"
tokenizer = Tokenizer.from_file(tokenizer_filepath)
padding_idx = tokenizer.encode("[PAD]").ids[0]
mask_idx = tokenizer.encode("[MASK]").ids[0]


def save_checkpoint(
    transformer,
    optimizer,
    epoch,
    filepaths,
    file_idx,
    checkpoint_dir,
    max_checkpoints=None,
):
    """Args:
    transformer (nn.Module): The model to save
    optimiser (torch.optim): The optimizer to save
    epoch (int): The current epoch
    filepaths (list): The ordered list of filepaths
    file_idx (int): The current file index
    checkpoint_dir (str): The directory to save the checkpoint
    elapsed (float): The time elapsed since training began
    loss (float): The mean masked likelihood
    max_checkpoints (int): The maximum number of checkpoints to keep
    """

    print(f"Saving checkpoint for epoch {epoch}, beginning with file {file_idx}...")

    # Remove old checkpoints if there are too many
    checkpoint_filepath = os.path.join(
        checkpoint_dir, f"epoch{epoch}_file{file_idx}.pt"
    )
    checkpoint_files = data_ops.gather_files(checkpoint_dir, file_extension=".pt")
    if max_checkpoints is not None:
        while len(checkpoint_files) > max_checkpoints - 1:
            checkpoint_files = sorted(
                checkpoint_files, key=lambda x: os.path.getctime(x)
            )
            oldest_file = checkpoint_files.pop(0)
            os.remove(oldest_file)

    # Save the checkpoint
    torch.save(
        {
            "epoch": epoch,
            "start_file_idx": file_idx,  # +1 to start from the next file
            "ordered_filepaths": filepaths,
            "model_state_dict": transformer.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        checkpoint_filepath,
    )


def load_checkpoint(checkpoint_name, checkpoint_dir, transformer, optimizer):
    """Load a checkpoint and return the epoch, file index, and ordered filepaths.
    Load model and optimizer in place, return the other values.

    Args:
        checkpoint_name (str): Name of the checkpoint file
        checkpoint_dir (str): Directory where the checkpoint is stored
        transformer (nn.Module): The model to load the checkpoint into
        optimizer (torch.optim): The optimizer to load the checkpoint into
    """
    if not os.path.exists(os.path.join(checkpoint_dir, checkpoint_name)):
        raise FileNotFoundError(f"Checkpoint file {checkpoint_name} not found")
    checkpoint = torch.load(os.path.join(checkpoint_dir, checkpoint_name))
    transformer.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return (
        checkpoint["epoch"],
        checkpoint["start_file_idx"],
        checkpoint["ordered_filepaths"],
    )


def write_loss(losses, loss_epoch_dir, loss_idx):
    if not os.path.exists(loss_epoch_dir):
        os.makedirs(loss_epoch_dir)
    with open(os.path.join(loss_epoch_dir, f"{loss_idx}.txt"), "w") as f:
        # Write the losses from a list to one line per loss
        for batch_loss in losses:
            f.write(f"{batch_loss[0]},{batch_loss[1]}\n")
    return


## Model Params
vocab_size = 15000
sequence_length = 64

# Model Parameters
d_embedding = 512
d_model = 512
d_ff = d_model * 4
n_layers = 8
dropout = 0.1

# k and v dims per head
d_k = 64
d_v = d_k
n_heads = d_model / d_k
if n_heads != int(n_heads):
    raise ValueError("d_model must be divisible by d_k")
n_heads = int(n_heads)


## Training Params
data_dir = "D:\\\\data\\embedded_text\\wikipedia_vocab64_seqlen15k"
batch_size = 128
n_epochs = 6
lr = 3e-4

mask_probability = 0.15

FLAG_LOAD_CHECKPOINT = True
checkpoint_dir = ".checkpoints"
checkpoint_epoch_dir = os.path.join(checkpoint_dir, "epoch")
checkpoint_every = 10
max_checkpoints = 5
checkpoint_name = "epoch0_file580.pt"

loss_dir = os.path.join(".metrics", "llh_loss")
loss_idx = 0

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
optimizer = optim.Adam(transformer.parameters())


if not FLAG_LOAD_CHECKPOINT:
    data_ops.create_directory(checkpoint_dir)
    data_ops.create_directory(checkpoint_epoch_dir)
    data_ops.create_directory(loss_dir)

    optimizer = optim.Adam(transformer.parameters(), lr=lr)

    start_epoch = 0
    file_idx = 0

    # Keep only the relative path from the data directory
    filepaths = data_ops.gather_files(data_dir, file_extension=".pt")
    np.random.shuffle(filepaths)
    rel_filepaths = [os.path.relpath(f, data_dir) for f in filepaths]

    checkpointed_files = 0
else:
    optimizer = optim.Adam(transformer.parameters())
    start_epoch, file_idx, rel_filepaths = load_checkpoint(
        checkpoint_name, checkpoint_dir, transformer, optimizer
    )
    initial_file_idx = file_idx
    checkpointed_files = file_idx
    filepaths = [os.path.join(data_dir, f) for f in rel_filepaths]


print("Training transformer model:")
print(transformer)
print(
    f"Number of parameters: {sum(p.numel() for p in transformer.parameters() if p.requires_grad)}\n"
)


losses = []
import time

start = time.time()
checkpoint_start = time.time()
for epoch in range(n_epochs):
    print(f"Epoch {epoch}/{n_epochs-1}")

    # Load data
    data_loader = data_ops.DataLoader(
        filepaths,
        batch_size=batch_size,
        file_idx=file_idx,
        shuffle_contents=True,
        device=device,
    )
    checkpoint_losses = []
    for i, batch, file_idx in data_loader:
        # Checkpoint
        if file_idx % checkpoint_every == 0 and file_idx > checkpointed_files:
            write_loss(checkpoint_losses, os.path.join(loss_dir, str(epoch)), loss_idx)
            loss_idx += 1
            save_checkpoint(
                transformer,
                optimizer,
                epoch,
                rel_filepaths,
                file_idx,
                checkpoint_dir,
                max_checkpoints=max_checkpoints,
            )
            elapsed = time.time() - start
            checkpoint_elapsed = time.time() - checkpoint_start
            mean_llh_str = f"{np.mean([x[1] for x in checkpoint_losses])}:.2f"
            print(
                f"Total: {data_ops.seconds_to_ms(elapsed)} | Checkpoint: {data_ops.seconds_to_ms(checkpoint_elapsed)} ({elapsed/(file_idx - initial_file_idx):.2f}s/file) | Mean Masked Likelihood: {mean_llh_str}"
            )
            checkpoint_start = time.time()
            checkpoint_losses = []
            checkpointed_files += checkpoint_every
        optimizer.zero_grad()

        # Mask tokens
        masked_batch, masked_batch_bool = data_ops.mask_tokens(
            batch,
            mask_id=mask_idx,
            pad_id=padding_idx,
            mask_prob=mask_probability,
            vocab_low_high=(5, vocab_size),
        )

        # Forward pass
        all_token_likelihoods = transformer(masked_batch)  # output: b,s,voc

        # Extract negative likelihoods of the masked tokens
        ground_truth_likelihoods = all_token_likelihoods.gather(
            -1, batch.unsqueeze(-1)
        ).squeeze()
        masked_ground_truth_likelihoods = ground_truth_likelihoods[masked_batch_bool]
        log_likelihood = torch.neg(torch.log(masked_ground_truth_likelihoods))

        # Backward pass
        loss = log_likelihood.mean()
        loss.backward()
        optimizer.step()
        checkpoint_losses.append((i, loss.item()))

    write_loss(checkpoint_losses, os.path.join(loss_dir, str(epoch)), loss_idx)
    loss_idx += 1
    save_checkpoint(
        transformer,
        optimizer,
        epoch,
        rel_filepaths,
        file_idx=0,
        checkpoint_dir=checkpoint_epoch_dir,
        max_checkpoints=None,
    )

    checkpointed_files = 0
