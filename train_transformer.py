import argparse
import os
import yaml
import time

import torch
import torch.optim as optim

from src import data_ops
from src import loss_functions
from src.checkpoints import load_checkpoint, save_checkpoint
from src.transformer import EncoderTransformer

from tokenizers import Tokenizer

import wandb


def main():

    ## Load tokenizer
    tokenizer = Tokenizer.from_file(TOKENIZER_FILEPATH)
    vocab_size = tokenizer.get_vocab_size()
    sequence_length = tokenizer.truncation["max_length"]
    padding_idx = tokenizer.encode("[PAD]").ids[0]
    mask_idx = tokenizer.encode("[MASK]").ids[0]

    ## LR Schedule
    lr_lambda = lambda step: TE_D_MODEL ** (-0.5) * min(
        (step + 1) ** (-0.5), (step + 1) * OPT_WARMUP_STEPS ** (-1.5)
    )
    # lr_lambda = lambda step: 2e-4

    ## Model
    transformer = EncoderTransformer(
        vocab_size,
        sequence_length,
        TE_N_LAYERS,
        TE_D_EMBEDDING,
        TE_D_MODEL,
        TE_D_K,
        TE_D_V,
        TE_N_HEADS,
        TE_D_FF,
        padding_idx,
        TE_DROPOUT,
    ).to(DEVICE)

    ## Initialise training objects - either from checkpoint or from scratch
    if FLAG_LOAD_CHECKPOINT:
        print(f"Loading checkpoint from {CHKPT_LOAD_FILEPATH}")
        optimizer = optim.Adam(transformer.parameters())
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        data_loader = data_ops.DataLoader(
            TRAIN_DATA_DIR, find_files=False, device=DEVICE
        )
        start_epoch = load_checkpoint(
            os.path.join(CHKPT_LOAD_FILEPATH),
            transformer,
            optimizer,
            scheduler,
            data_loader,
        )
        file_idx = data_loader.file_idx
        initial_file_idx = file_idx
        last_checkpoint_idx = file_idx
    else:
        data_ops.create_directory(CHKPT_DIR, reset=True)
        data_ops.create_directory(CHKPT_EPOCH_DIR, reset=True)

        optimizer = optim.Adam(
            transformer.parameters(), weight_decay=OPT_WEIGHT_DECAY, lr=OPT_LR_SCALE
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        start_epoch = 0
        file_idx = 0
        initial_file_idx = 0

        # Keep only the relative path from the data directory
        data_loader = data_ops.DataLoader(
            TRAIN_DATA_DIR, TRAIN_BATCH_SIZE, device=DEVICE
        )
        last_checkpoint_idx = 0

    if FLAG_TRACK_WANDB:
        wandb.init(
            project="transformer-encoder",
            run_id=WANDB_RUN_ID if WANDB_RUN_ID else wandb.util.generate_id(),
            resume="must" if WANDB_RUN_ID else None,
            name=RUN_NAME,
            config=dict(
                {
                    "dataset": TRAIN_DATA_DIR,
                    "tokenizer": TOKENIZER_FILEPATH,
                    "vocab_size": vocab_size,
                    "sequence_length": sequence_length,
                },
                # Extract the values from the config dictionary
                **{k: v["value"] for k, v in CONFIG_DICT.items()},
            ),
        )
    weight_tracking_startswith = tuple(["embedding", "mhsa.mhsa_0", "output"])

    print("Training transformer model:")
    print(transformer)
    print(
        f"Number of parameters: {sum(p.numel() for p in transformer.parameters() if p.requires_grad)}\n"
    )
    end_run = False

    for epoch in range(start_epoch, TRAIN_N_EPOCHS):
        print(f"Epoch {epoch}/{TRAIN_N_EPOCHS-1}")
        start = time.time()
        checkpoint_start = time.time()

        for batch, batch_counter, file_idx in data_loader:
            optimizer.zero_grad()

            ## Checkpoint
            if file_idx % CHKPT_RECORD_EVERY == 0 and file_idx > last_checkpoint_idx:
                save_checkpoint(
                    CHKPT_DIR,
                    transformer,
                    optimizer,
                    scheduler,
                    data_loader,
                    RUN_NAME,
                    epoch,
                    file_idx,
                    CHKPT_MAX_CHECKPOINTS,
                )
                print_progress(
                    start,
                    checkpoint_start,
                    file_idx - initial_file_idx,
                    file_idx - last_checkpoint_idx,
                )
                checkpoint_start = time.time()
                last_checkpoint_idx = file_idx

            ## Mask tokens
            batch_masked, batch_masked_bool, batch_attention_mask = (
                data_ops.mask_tokens(
                    batch,
                    mask_id=mask_idx,
                    pad_id=padding_idx,
                    mask_prob=MASK_PROBABILITY,
                    vocab_low_high=(5, vocab_size),
                    proportion_mask_token=MASK_PROPORTION_MASK_TAKEN,
                    proportion_random_token=MASK_PROPORTION_RANDOM_TOKEN,
                )
            )

            ## Forward pass
            batch_all_token_outputs = transformer(
                batch_masked, batch_attention_mask
            )  # unnormailzed logits

            ## Calculate loss
            loss = loss_functions.cross_entropy(
                batch_all_token_outputs, batch, batch_masked_bool
            )

            ## Optimize
            loss.backward()
            optimizer.step()
            scheduler.step()

            ## Track
            if FLAG_TRACK_WANDB:
                grad_norms = {}
                weight_stats = {}
                for name, param in transformer.named_parameters():
                    if name.startswith(weight_tracking_startswith):
                        if param.grad is not None:
                            grad_norms[name] = param.grad.data.norm(2).item()
                        weight_stats[name + "_mean"] = param.data.mean().item()
                        weight_stats[name + "_std"] = param.data.std().item()

                wandb.log(
                    {
                        "epoch": epoch,
                        "batch_number": batch_counter,
                        "cross_entropy": loss.item(),
                        "learning_rate": scheduler.get_last_lr()[0],
                        "weight_stats": weight_stats,
                        "grad_norms": grad_norms,
                    }
                )

            # If loss.item() is nan, record diagnostics to file and break
            if torch.isnan(loss):
                print("Loss is NaN")
                end_run = True
                break

        if end_run:
            break

        ## Reset loaded variables
        initial_file_idx = 0
        last_checkpoint_idx = -1  # to ensure the first checkpoint is saved

    ## Save final model
    save_checkpoint(
        CHKPT_EPOCH_DIR,
        transformer,
        optimizer,
        scheduler,
        data_loader,
        RUN_NAME,
        epoch + 1,
        file_idx=0,
        max_checkpoints=CHKPT_MAX_CHECKPOINTS,
    )
    print_progress(
        start,
        checkpoint_start,
        file_idx - initial_file_idx,
        file_idx - last_checkpoint_idx,
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


if __name__ == "__main__":

    ## Parse command-line arguments
    parser = argparse.ArgumentParser(description="Transformer Training Script")
    parser.add_argument(
        "--config_file",
        type=str,
        default="model-config.yaml",
        help="Config YAML file name",
    )
    parser.add_argument(
        "--config_dir",
        type=str,
        default="configs",
        help="Directory containing config YAML file",
    )
    parser.add_argument(
        "--data_dir", type=str, required=True, help="Directory for training data"
    )
    parser.add_argument(
        "--tokenizer_filepath", type=str, required=True, help="Path to tokenizer file"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=".checkpoints",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--load_checkpoint",
        type=str,
        default="",
        help="Path to a specific checkpoint to load, leave empty to start from scratch",
    )
    parser.add_argument(
        "--checkpoint_every",
        type=int,
        default=50,
        help="Number of batches between each checkpoint",
    )
    parser.add_argument(
        "--max_checkpoints",
        type=int,
        default=5,
        help="Maximum number of checkpoints to keep",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Track training progress with Weights and Biases",
    )
    parser.add_argument(
        "wandb_run_id", type=str, default="", help="WandB run ID, set to resume run"
    )
    args = parser.parse_args()

    ## Load configuration from YAML file
    with open(os.path.join(args.config_dir, args.config_file), "r") as file:
        CONFIG_DICT = yaml.safe_load(file)

    ## Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    ## Tokenizer
    TOKENIZER_FILEPATH = args.tokenizer_filepath

    ## Model
    TE_D_MODEL = CONFIG_DICT["d_model"]["value"]
    TE_D_EMBEDDING = CONFIG_DICT["d_embedding"]["value"]
    TE_D_FF = CONFIG_DICT["d_ff"]["value"]
    TE_N_LAYERS = CONFIG_DICT["n_layers"]["value"]
    TE_DROPOUT = CONFIG_DICT["dropout"]["value"]

    TE_D_K = CONFIG_DICT["d_k"]["value"]  # k and v dims per head
    TE_D_V = CONFIG_DICT["d_v"]["value"]
    TE_N_HEADS = CONFIG_DICT["n_heads"]["value"]

    ## Training
    TRAIN_DATA_DIR = args.data_dir
    # "D://data/embedded_text/wikipedia_vocab64_seqlen15k/train"
    # data_dir = ".data\\tokenized_test_128"
    TRAIN_BATCH_SIZE = CONFIG_DICT["batch_size"]["value"]
    TRAIN_N_EPOCHS = CONFIG_DICT["epochs"]["value"]

    ## Masking
    MASK_PROBABILITY = CONFIG_DICT["mask_probability"]["value"]
    MASK_PROPORTION_MASK_TAKEN = CONFIG_DICT["proportion_mask_token"]["value"]
    MASK_PROPORTION_RANDOM_TOKEN = CONFIG_DICT["proportion_random_token"]["value"]

    ## Learning rate scheduler
    OPT_WARMUP_STEPS = CONFIG_DICT["warmup_steps"]["value"]
    OPT_LR_SCALE = CONFIG_DICT["lr_scale"]["value"]
    OPT_WEIGHT_DECAY = CONFIG_DICT["weight_decay"]["value"]

    ## Checkpointing
    FLAG_LOAD_CHECKPOINT = args.load_checkpoint != ""
    CHKPT_DIR = args.checkpoint_dir
    CHKPT_EPOCH_DIR = os.path.join(CHKPT_DIR, "epoch")
    CHKPT_RECORD_EVERY = args.checkpoint_every
    CHKPT_MAX_CHECKPOINTS = args.max_checkpoints
    CHKPT_LOAD_FILEPATH = args.load_checkpoint

    ## WandB
    FLAG_TRACK_WANDB = args.wandb
    WANDB_RUN_ID = args.wandb_run_id

    ## Run name
    run_identifier = args.config_file.split("/")[-1].split(".")[0]
    # run_identifier = f"lrs{OPT_LR_SCALE}_wu{OPT_WARMUP_STEPS/1000:.1f}K".replace(".", "-")
    run_suffix = time.strftime("%m%d_%H%M")
    RUN_NAME = f"{run_identifier}_{run_suffix}"

    main()
