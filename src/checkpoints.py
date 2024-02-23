import torch
import os
from . import data_ops


def save_checkpoint(
    checkpoint_dir,
    transformer,
    optimizer,
    data_loader,
    epoch,
    file_idx,
    max_checkpoints=None,
):
    """Args:
    transformer (nn.Module): The model to save
    optimizer (torch.optim): The optimizer to save
    epoch (int): The epoch number
    filepaths (list): The ordered list of filepaths
    file_idx (int): The index of the file to start with
    checkpoint_dir (str): The directory to save the checkpoint
    max_checkpoints (int): The maximum number of checkpoints to keep
    """

    checkpoint_filepath = os.path.join(
        checkpoint_dir, f"epoch{epoch}_file{file_idx}.pt"
    )
    print(
        f"Saving checkpoint for epoch {epoch}, beginning with file {file_idx} to {checkpoint_filepath}"
    )

    # Remove old checkpoints if there are too many
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
            "model_state_dict": transformer.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "dataloader_state_dict": data_loader.state_dict(),
        },
        checkpoint_filepath,
    )

    print(f"Checkpoint saved")


def load_checkpoint(filepath, transformer, optimizer, data_loader):
    """Load a checkpoint and return the epoch, file index, and ordered filepaths.
    Load model and optimizer in place, return the other values.

    Args:
        checkpoint_name (str): Name of the checkpoint file
        checkpoint_dir (str): Directory where the checkpoint is stored
        transformer (nn.Module): The model to load the checkpoint into
        optimizer (torch.optim): The optimizer to load the checkpoint into
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint file {filepath} not found")
    checkpoint = torch.load(filepath)
    transformer.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    data_loader.load_state_dict(checkpoint["dataloader_state_dict"])
    return checkpoint["epoch"]
