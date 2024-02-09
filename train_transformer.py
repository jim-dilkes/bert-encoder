
import os
import time
import torch
import torch.optim as optim
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import data_ops
from transformer import EncoderTransformer

import numpy as np

# load tokenizer
from tokenizers import Tokenizer
tokenizer = Tokenizer.from_file("tokenizer_V2.json")
padding_idx = tokenizer.encode('[PAD]').ids[0]
mask_idx = tokenizer.encode('[MASK]').ids[0]


## Model Params
vocab_size = 15000
sequence_length = 128

# Model Parameters
d_embedding = 256
d_model = 256
d_ff = d_model * 4
n_layers = 4

# k and v dims per head
d_k = 64
d_v = d_k
n_heads = d_model / d_k
if n_heads != int(n_heads):
    raise ValueError("d_model must be divisible by d_k")
n_heads = int(n_heads)


## Training Params
root_dir = 'data_tokenized'
batch_size = 150
n_epochs = 6
learning_rates = [0.001, 3e-4, 1e-4]

mask_probability = 0.15

checkpoint_dir = 'checkpoints'
checkpoint_every = 1000
max_checkpoints = 5
checkpoint_name = 'checkpoint_epoch0_batch23999.pt'

loss_dir = 'losses'
loss_idx = 0

restart_training = True
if restart_training:
    checkpoint_name=None
    data_ops.reset_directory(checkpoint_dir)
    data_ops.reset_directory(loss_dir)


def checkpoint(transformer, epoch, batch, checkpoint_dir, elapsed, loss, max_checkpoints=5):
    print(f'\nSaving checkpoint... \nTime Elapsed: {elapsed/60:.2f} | Batch {i+1} ({(elapsed/(i+1)):.2f}) | Mean Masked Likelihood: {loss*-1}')
    torch.save(transformer.state_dict(), f'{checkpoint_dir}\\checkpoint_epoch{epoch}_batch{batch+1}.pt')
    checkpoint_files = data_ops.gather_files(checkpoint_dir, file_extension='.pt')
    while len(checkpoint_files) > max_checkpoints:
        checkpoint_files = sorted(checkpoint_files, key=lambda x: os.path.getctime(x), reverse=True)
        oldest_file = checkpoint_files.pop(0)
        os.remove(oldest_file)
    return


def write_loss(losses, loss_epoch_dir, loss_idx):
    if not os.path.exists(loss_epoch_dir):
        os.makedirs(loss_epoch_dir)
    with open(f'{loss_epoch_dir}\\{loss_idx}.txt', 'w') as f:
        # Write the losses from a list to one line per loss
        for batch_loss in losses:
            f.write(f"{batch_loss[0]},{batch_loss[1]}\n")
    return


filepaths = data_ops.gather_files(root_dir, file_extension='.pt')
np.random.shuffle(filepaths)


transformer = EncoderTransformer(vocab_size, sequence_length, n_layers, d_embedding, d_model, d_k, d_v, n_heads, d_ff, padding_idx).to(device)
if checkpoint_name is not None:
    transformer.load_state_dict(torch.load(checkpoint_name))

print("Training transformer model:")
print(transformer)
print(f"\nNumber of parameters:{sum(p.numel() for p in transformer.parameters() if p.requires_grad)}\n")


losses = []
import time
start = time.time()
for epoch in range(n_epochs):
    print(f"\nEpoch {epoch}/{n_epochs-1}")

    # Set learning rate for this epoch
    if len(learning_rates) > 1:
        lr = learning_rates.pop(0)
    else:
        lr = learning_rates[0]
    optimizer = optim.Adam(transformer.parameters(), lr=lr)

    # Load data
    data_loader = data_ops.DataLoader(filepaths, batch_size=batch_size, shuffle_contents=True, device=device)
    epoch_losses = []
    for i, batch in enumerate(data_loader):
        optimizer.zero_grad()

        # Mask tokens
        masked_batch, masked_batch_bool = data_ops.mask_tokens(batch, mask_id=mask_idx, pad_id=padding_idx, mask_prob=mask_probability, vocab_low_high=(5,vocab_size))

        # Forward pass
        all_token_likelihoods = transformer(masked_batch) # output: b,s,voc
        
        # Extract negative likelihoods of the masked tokens
        ground_truth_likelihoods = all_token_likelihoods.gather(-1, batch.unsqueeze(-1)).squeeze()
        masked_ground_truth_likelihoods = torch.neg(ground_truth_likelihoods[masked_batch_bool])

        # Backward pass
        loss = masked_ground_truth_likelihoods.mean()
        loss.backward()
        optimizer.step()
        epoch_losses.append((i, loss.item()))

        # Checkpoint
        if i % checkpoint_every == checkpoint_every-1:
            write_loss(epoch_losses, f"{loss_dir}\\{epoch}", loss_idx)
            loss_idx += 1
            epoch_losses = []
            checkpoint(transformer, epoch, i, checkpoint_dir, time.time()-start, loss.item(), max_checkpoints=max_checkpoints)

    write_loss(epoch_losses, f"{loss_dir}\\{epoch}", loss_idx)
    loss_idx += 1
    checkpoint(transformer, epoch, i, checkpoint_dir, time.time()-start, loss.item(), max_checkpoints=max_checkpoints)