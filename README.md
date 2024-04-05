# Transformer Encoder Training README

## Purpose
This script is designed for training an encoder-only transformer model with the [BERT](https://arxiv.org/abs/1810.04805) architecture, using improvements from [RoBERTa](https://arxiv.org/abs/1907.11692).
It supports flexible configurations through a YAML file, batch processing with asynchronous data loading, checkpoint management, and integration with Weights & Biases for tracking and logging training progress.

## How to Use
1. Prepare your tokenizer - uses the Hugging Face [tokenizers](https://huggingface.co/docs/tokenizers/) library.
2. Prepare your data.

    a. Pre-tokenize the data and store it as .pt files containing only the tensor of tokenized data for some number of examples.
    
    b. The script will train on all examples in every .pt files in the directory passed to --data_dir and its subdirectories.
3. Create a YAML config file containing model definition and training parameters.
4. Run the script with required arguments, specifying paths to your data, tokenizer, and configuration file.
   
    a. Optionally, enable [Weights & Biases](https://wandb.ai/) tracking.
    

## Command-Line Arguments
- `--config_file`: Name of the YAML configuration file with model and training settings.
- `--config_dir`: Directory containing the YAML configuration file.
- `--data_dir`: Directory for training data.
- `--tokenizer_filepath`: Path to the tokenizer file.
- `--checkpoint_dir`: Directory to save training checkpoints.
- `--load_checkpoint`: Path to a specific checkpoint file to resume training.
- `--checkpoint_every`: Frequency (in batches) of checkpoint saving.
- `--max_checkpoints`: Maximum number of checkpoints to retain.
- `--wandb`: Enable tracking with Weights & Biases.
- `--wandb_run_id`: Weights & Biases run ID for resuming tracking.
- `--wandb_log_freq`: Frequency (in batches) of logging to Weights & Biases.
- `--wandb_project_name`: The name of the Weights & Biases project you want to record to.
- `--batch_size`: Batch size for training (overrides config file).
- `--gradient_clip`: Gradient clipping value (overrides config file).
