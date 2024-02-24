import torch
import os
from . import data_ops


def save_checkpoint(
    checkpoint_dir,
    transformer,
    optimizer,
    scheduler,
    data_loader,
    epoch,
    file_idx,
    max_checkpoints=None,
):
    """Save a checkpoint to a file

    Args:
    checkpoint_dir (str): Directory to save the checkpoint
    transformer (nn.Module): The model to save
    optimizer (torch.optim): The optimizer to save
    scheduler (torch.optim.lr_scheduler): The scheduler to save
    data_loader (data_ops.DataLoader): The data loader to save
    epoch (int): The epoch to save
    file_idx (int): The file index to save
    max_checkpoints (int): The maximum number of checkpoints to keep in the directory
    """

    checkpoint_filepath = os.path.join(
        checkpoint_dir, f"epoch{epoch}_file{file_idx}.pt"
    )
    print(
        f"Saving checkpoint prior to file {file_idx} in epoch {epoch} to {checkpoint_filepath}"
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
            "scheduler_state_dict": scheduler.state_dict(),
            "dataloader_state_dict": data_loader.state_dict(),
        },
        checkpoint_filepath,
    )

    print(f"Checkpoint saved")


def load_checkpoint(filepath, transformer, optimizer, scheduler, data_loader):
    """Load a checkpoint from a file

    Args:
    filepath (str): The filepath to the checkpoint
    transformer (nn.Module): The model to load in to
    optimizer (torch.optim): The optimizer to load in to
    scheduler (torch.optim.lr_scheduler): The scheduler to load in to
    data_loader (data_ops.DataLoader): The data loader to load in to
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint file {filepath} not found")
    checkpoint = torch.load(filepath)
    transformer.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    data_loader.load_state_dict(checkpoint["dataloader_state_dict"])
    return checkpoint["epoch"]
