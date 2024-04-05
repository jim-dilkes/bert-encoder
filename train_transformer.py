import argparse
import os
from math import exp
import yaml
import time

import torch
import torch.optim as optim

from src import data_ops
from src import loss_functions
from src.checkpoints import load_checkpoint, save_checkpoint
from src.asynchronousbatchloader import AsynchronousBatchLoader
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
        dropout_ff=TE_DROPOUT,
        dropout_mhsa=0,
        use_custom_mhsa=True,
    ).to(DEVICE)
    n_params = sum(p.numel() for p in transformer.parameters() if p.requires_grad)

    ## Initialise training objects - either from checkpoint or from scratch
    if FLAG_LOAD_CHECKPOINT:
        print(f"Loading checkpoint from {CHKPT_LOAD_FILEPATH}")
        optimizer = optim.Adam(transformer.parameters())
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        data_loader = AsynchronousBatchLoader(
            TRAIN_DATA_DIR, TRAIN_BATCH_SIZE, device=DEVICE  # , debug=True
        )
        counters = load_checkpoint(
            os.path.join(CHKPT_LOAD_FILEPATH),
            transformer,
            optimizer,
            scheduler,
            data_loader,
        )
        initial_epoch = counters["epoch"]
        initial_n_batches = counters["global_batches"]
        last_checkpoint_n_batches = counters["global_batches"]
        initial_n_examples = counters["global_train_examples"]
        last_checkpoint_n_examples = counters["global_train_examples"]

        FLAG_RESCALE_LR = True
        if FLAG_RESCALE_LR:
            scale_factor = 0.3
            print("Rescaling learning rate by ", scale_factor)
            for param_group in optimizer.param_groups:
                param_group["lr"] = OPT_LR_SCALE

    else:
        data_ops.create_directory(CHKPT_DIR, reset=True)
        data_ops.create_directory(CHKPT_EPOCH_DIR, reset=True)

        optimizer = optim.Adam(
            transformer.parameters(), weight_decay=OPT_WEIGHT_DECAY, lr=OPT_LR_SCALE
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        initial_epoch = 0
        initial_n_batches = 0
        last_checkpoint_n_batches = 0
        initial_n_examples = 0
        last_checkpoint_n_examples = 0

        max_file_size = 10_000
        batch_per_file = max_file_size // TRAIN_BATCH_SIZE
        data_loader = AsynchronousBatchLoader(
            TRAIN_DATA_DIR,
            TRAIN_BATCH_SIZE,
            device=DEVICE,
            max_queue_size=3 * batch_per_file,  # Load max 3 files of batches queued
            # debug=True,
        )

        counters = {
            "epoch": 0,
            "epoch_batches": 0,
            "global_batches": 0,
            "epoch_train_examples": 0,
            "global_train_examples": 0,
        }

    if TRAIN_OVERRIDE_BATCH_SIZE is not None:
        data_loader.batch_size = TRAIN_OVERRIDE_BATCH_SIZE

    if FLAG_TRACK_WANDB:
        if FLAG_RESUME_WANDB:
            wandb.init(project=PROJECT_NAME, id=WANDB_RUN_ID, resume="must")
        else:
            wandb.init(
                project=PROJECT_NAME,
                id=WANDB_RUN_ID,
                name=RUN_NAME,
                config=dict(
                    {
                        "dataset": TRAIN_DATA_DIR,
                        "tokenizer": TOKENIZER_FILEPATH,
                        "vocab_size": vocab_size,
                        "sequence_length": sequence_length,
                        "model_config": {k: v["value"] for k, v in CONFIG_DICT.items()},
                        "model_n_params": n_params,
                    },
                ),
            )
    weight_tracking_startswith = tuple(["embedding", "mhsa.mhsa_0", "output"])

    print("Training transformer model:")
    print(transformer)
    print(f"Number of parameters: {n_params}\n")

    for _ in range(initial_epoch, TRAIN_N_EPOCHS):
        print(f"Epoch {counters['epoch']+1}/{TRAIN_N_EPOCHS} (idx {counters['epoch']})")
        start = time.time()
        checkpoint_start = time.time()

        for batch, this_batch_size in data_loader:
            # print(f"Batch {i} received with {this_batch_size} examples")
            optimizer.zero_grad()

            ## Mask tokens
            batch_masked, batch_masked_bool, batch_attention_mask = (
                data_ops.mask_tokens(
                    batch,
                    mask_id=mask_idx,
                    pad_id=padding_idx,
                    mask_prob=MASK_PROBABILITY,
                    vocab_low_high=(5, vocab_size),
                    proportion_mask_token=MASK_PROPORTION_MASK_TOKEN,
                    proportion_random_token=MASK_PROPORTION_RANDOM_TOKEN,
                )
            )

            ## Forward pass
            batch_all_token_outputs = transformer(
                batch_masked, batch_attention_mask
            )  # unnormalized logits

            ## Calculate loss
            loss = loss_functions.cross_entropy(
                batch_all_token_outputs, batch, batch_masked_bool
            )
            ## Optimize
            loss.backward()
            if TRAIN_GRADIENT_CLIP is not None:
                torch.nn.utils.clip_grad_norm_(
                    transformer.parameters(), TRAIN_GRADIENT_CLIP
                )
            optimizer.step()
            scheduler.step()

            # print(f"Batch {i} processed")

            ## Track
            if counters["global_batches"] % WANDB_LOG_FREQ == 0:
                print_progress(
                    start,
                    checkpoint_start,
                    counters["global_batches"] - initial_n_batches,
                    counters["global_batches"] - last_checkpoint_n_batches,
                    counters["global_train_examples"] - initial_n_examples,
                    counters["global_train_examples"] - last_checkpoint_n_examples,
                    counters["global_batches"],
                    counters["global_train_examples"],
                )
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
                            "epoch": counters["epoch"],
                            "epoch_batches_seen": counters["epoch_batches"],
                            "global_batches_seen": counters["global_batches"],
                            "epoch_train_examples_seen": counters[
                                "epoch_train_examples"
                            ],
                            "global_train_examples_seen": counters[
                                "global_train_examples"
                            ],
                            "cross_entropy": loss.item(),
                            "probability": exp(-loss.item()),
                            "learning_rate": scheduler.get_last_lr()[0],
                            "weight_stats": weight_stats,
                            "grad_norms": grad_norms,
                        }
                    )

            counters["epoch_batches"] += 1
            counters["global_batches"] += 1
            counters["epoch_train_examples"] += this_batch_size
            counters["global_train_examples"] += this_batch_size

            ## Checkpoint
            if (
                counters["global_batches"] % CHKPT_RECORD_EVERY == 0
                and counters["global_batches"] > last_checkpoint_n_batches
            ):
                save_checkpoint(
                    CHKPT_DIR,
                    counters,
                    transformer,
                    optimizer,
                    scheduler,
                    data_loader,
                    RUN_NAME,
                    CHKPT_MAX_CHECKPOINTS,
                )
                # print_progress(
                #     start,
                #     checkpoint_start,
                #     counters["global_batches"] - initial_n_batches,
                #     counters["global_batches"] - last_checkpoint_n_batches,
                #     counters["global_train_examples"],
                #     counters["global_train_examples"] - last_checkpoint_n_examples,
                # )
                checkpoint_start = time.time()
                last_checkpoint_n_batches = counters["global_batches"]
                last_checkpoint_n_examples = counters["global_train_examples"]

        ## Save epoch model
        print(
            f"Epoch {counters['epoch'] + 1} of {TRAIN_N_EPOCHS} complete (idx {counters['epoch']}), saving checkpoint for next epoch start"
        )
        counters["epoch"] += 1
        counters["epoch_batches"] = 0
        counters["epoch_train_examples"] = 0
        save_checkpoint(
            CHKPT_EPOCH_DIR,
            counters,
            transformer,
            optimizer,
            scheduler,
            data_loader,
            RUN_NAME,
            CHKPT_MAX_CHECKPOINTS,
        )


def print_progress(
    total_start,
    checkpoint_start,
    this_session_n_batches,
    checkpoint_n_batches,
    this_session_n_examples,
    checkpoint_n_examples,
    total_n_batches,
    total_n_examples,
):
    if this_session_n_batches == 0:
        return
    total_elapsed = time.time() - total_start
    checkpoint_elapsed = time.time() - checkpoint_start
    total_avg_time = total_elapsed / this_session_n_batches
    checkpoint_avg_time = (
        checkpoint_elapsed / checkpoint_n_batches if checkpoint_n_batches > 0 else 0
    )
    total_ex_s = int(this_session_n_examples / total_elapsed)
    checkpoint_ex_s = int(checkpoint_n_examples / checkpoint_elapsed)

    print(
        f"Done {total_n_batches} batches / {total_n_examples} examples total, {this_session_n_batches} batches / {this_session_n_examples} examples this session, {checkpoint_n_batches} batches / {checkpoint_n_examples} examples this checkpoint | Session time: {data_ops.seconds_to_ms(total_elapsed)} ({total_avg_time:.4f}s/batch, {total_ex_s}ex/s) | Checkpoint time: {data_ops.seconds_to_ms(checkpoint_elapsed)} ({checkpoint_avg_time:.4f}s/batch, {checkpoint_ex_s}ex/s)"
    )


def check_nans(model, output, loss):
    flag_nan = False
    if torch.isnan(output).any():
        n_nan = torch.isnan(output).sum().item()
        n_total = output.numel()
        print(f"Output is NaN {n_nan/n_total} ({n_nan}/{n_total})")
        flag_nan = True
    if torch.isnan(loss):
        print("Loss is NaN")
        flag_nan = True
    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad.data).any():
                n_nan = torch.isnan(param.grad.data).sum().item()
                n_total = param.grad.data.numel()
                print(f"Gradient is NaN for {name} {n_nan/n_total} ({n_nan}/{n_total})")
                flag_nan = True
        if torch.isnan(param.data).any():
            n_nan = torch.isnan(param.data).sum().item()
            n_total = param.data.numel()
            print(f"Param is NaN for {name} {n_nan/n_total} ({n_nan}/{n_total})")
            flag_nan = True

    return flag_nan


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
        default=1000,
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
        "--wandb_run_id", type=str, default="", help="WandB run ID, set to resume run"
    )
    parser.add_argument(
        "--batch_size", type=int, default=None, help="Override the config batch size"
    )
    parser.add_argument(
        "--gradient_clip",
        type=float,
        default=None,
        help="Override the config gradient clip value",
    )
    parser.add_argument(
        "--wandb_log_freq",
        type=int,
        default=100,
        help="Number of batches between each log entry",
    )
    parser.add_argument(
        "--wandb_project_name", type=str, default="", help="WandB project name"
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
    TRAIN_OVERRIDE_BATCH_SIZE = args.batch_size
    TRAIN_N_EPOCHS = CONFIG_DICT["epochs"]["value"]
    TRAIN_GRADIENT_CLIP = (
        args.gradient_clip
        if args.gradient_clip
        else (
            CONFIG_DICT["gradient_clip"]["value"]
            if "gradient_clip" in CONFIG_DICT
            else None
        )
    )

    ## Masking
    MASK_PROBABILITY = CONFIG_DICT["mask_probability"]["value"]
    MASK_PROPORTION_MASK_TOKEN = CONFIG_DICT["proportion_mask_token"]["value"]
    MASK_PROPORTION_RANDOM_TOKEN = CONFIG_DICT["proportion_random_token"]["value"]

    ## Learning rate scheduler
    OPT_WARMUP_STEPS = CONFIG_DICT["warmup_steps"]["value"]
    OPT_LR_SCALE = CONFIG_DICT["lr_scale"]["value"]
    OPT_WEIGHT_DECAY = (
        CONFIG_DICT["weight_decay"]["value"] if "weight_decay" in CONFIG_DICT else 0
    )
    OPT_EPS = CONFIG_DICT["eps"]["value"] if "eps" in CONFIG_DICT else 1e-8

    ## Checkpointing
    FLAG_LOAD_CHECKPOINT = args.load_checkpoint != ""
    CHKPT_DIR = args.checkpoint_dir
    CHKPT_EPOCH_DIR = os.path.join(CHKPT_DIR, "epoch")
    CHKPT_RECORD_EVERY = args.checkpoint_every
    CHKPT_MAX_CHECKPOINTS = args.max_checkpoints
    CHKPT_LOAD_FILEPATH = args.load_checkpoint

    ## WandB
    FLAG_TRACK_WANDB = args.wandb
    FLAG_RESUME_WANDB = args.wandb_run_id != ""
    PROJECT_NAME = args.wandb_project_name
    WANDB_RUN_ID = (
        args.wandb_run_id
        if args.wandb_run_id
        else wandb.util.generate_id() if FLAG_TRACK_WANDB else ""
    )
    WANDB_LOG_FREQ = args.wandb_log_freq

    ## Run name
    run_identifier = args.config_file.split("/")[-1].split(".")[0]
    # run_identifier = f"lrs{OPT_LR_SCALE}_wu{OPT_WARMUP_STEPS/1000:.1f}K".replace(".", "-")
    run_time = time.strftime("%m%d_%H%M")
    RUN_NAME = (
        f"{run_identifier}_{run_time}{'_' + WANDB_RUN_ID if WANDB_RUN_ID else ''}"
    )

    main()
