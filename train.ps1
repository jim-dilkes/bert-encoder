# Define the arguments
$configFile = "--config_file 4_128-seq64-lrs1_0-4head.yaml"
# $configFile = "--config_file 6_256-seq64-lrs1_4.yaml"
# $configFile = "--config_file med-seq64_lrs1_clip1.yaml"
# $configFile = "--config_file med-seq64_lrs1_epsE06.yaml"

# $dataDir = "--data_dir D:\data\embedded_text\wikipedia_vocab64_seqlen15k\train"
$dataDir = "--data_dir D:\workspace\transformerencoder\.data\tokenized_test_128_V2"

$tokenizerFilepath = "--tokenizer_filepath .tokenizers\tok_SL64_V15000.json"

# $checkpointFile = "--load_checkpoint .archive\6_256-seq64-lrs1_4_0316_1642_xicyry2e_epoch0_file5700.pt"
# $checkpointFile = ""

# $wandb = 1
# $wandb_run_id = "xicyry2e"
$wandb_log_freq = "--wandb_log_freq 1000"

$checkpoint_every = "--checkpoint_every 10000"



# Build the command
$command = "python .\train_transformer.py"

if ($configFile) {
    $command += " $configFile"
}
if ($dataDir) {
    $command += " $dataDir"
}
if ($tokenizerFilepath) {
    $command += " $tokenizerFilepath"
}
if ($checkpointFile) {
    $command += " $checkpointFile"
}
if ($checkpoint_every) {
    $command += " $checkpoint_every"
}
if ($wandb) {
    $command += " --wandb"
}

if ($wandb_run_id) {
    $command += " --wandb_run_id $wandb_run_id"
}

if ($wandb_log_freq) {
    $command += " $wandb_log_freq"
}

# Print the command
Write-Host $command

# Execute the command
# Invoke-Expression $command