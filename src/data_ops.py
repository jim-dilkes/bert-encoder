import os
import shutil
import torch
from numpy.random import default_rng
from einops import rearrange


def seconds_to_ms(seconds: float) -> str:
    """Convert seconds to a string in the format 'mm:ss'."""
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"


def create_directory(directory: str, reset: bool = False):
    """Delete and recreate a directory"""
    if os.path.exists(directory):
        if reset:
            shutil.rmtree(directory)  # Delete the directory and its contents
            os.makedirs(directory)
    else:
        os.makedirs(directory)


def gather_files(directory: str, file_extension: str = None) -> list[str]:
    """Gather a list of filepaths from a directory and its subdirectories. Optionally filter by file_extension."""
    file_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file_extension is None or file[-len(file_extension) :] == file_extension:
                file_list.append(os.path.join(root, file))
    return file_list


def mask_tokens(
    input_batch: torch.Tensor,
    mask_id: int,
    pad_id: int,
    mask_prob: float = 0.1,
    vocab_low_high: tuple[int, int] = (5, 15000),
    proportion_mask_token: float = 0.8,
    proportion_random_token: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mask random tokens in a batch of input data."""

    assert mask_prob <= 1.0 and mask_prob >= 0
    use_device = input_batch.device

    ## Select tokens for masking
    mask_probabilities = torch.full(input_batch.shape, mask_prob, device=use_device)
    padding_mask = input_batch.eq(pad_id)
    mask_probabilities.masked_fill_(padding_mask, 0)
    masked_tokens_bool = torch.bernoulli(mask_probabilities).bool()

    ## Replace selected tokens
    masked_input_batch = input_batch.clone()

    # replace masked input tokens with mask_id
    replaced_tokens_bool = (
        torch.bernoulli(
            torch.full(input_batch.shape, proportion_mask_token, device=use_device)
        ).bool()
        & masked_tokens_bool
    )
    masked_input_batch.masked_fill_(replaced_tokens_bool, mask_id)

    # replace masked input tokens with random word
    random_tokens_bool = (
        torch.bernoulli(
            torch.full(
                input_batch.shape,
                proportion_random_token / (1 - proportion_mask_token),
                device=use_device,
            )
        ).bool()
        & masked_tokens_bool
        & ~replaced_tokens_bool
    )
    random_words = torch.randint(
        vocab_low_high[0],
        vocab_low_high[1],
        input_batch.shape,
        dtype=torch.long,
        device=use_device,
    )
    masked_input_batch[random_tokens_bool] = random_words[random_tokens_bool]

    attention_mask = (~padding_mask).long()

    return masked_input_batch, masked_tokens_bool, attention_mask


class DataLoader:
    """
    DataLoader for loading and iterating through batches of data from a list of filepaths.
    - Loads data from filepaths in order, splitting each file into batches of size batch_size
    - Once a file is exhausted, it loads the next file in the list
    - If examples dont divide evenly into batches, the remaining examples are held in a hanging batch
    - The hanging batch is prepended to the next file's data
    - If there is not enough data to fill a final batch, the final hanging batch is ignored to avoid a partial batch
    """

    def __init__(
        self,
        root_directory: str,
        batch_size: int = None,
        find_files: bool = True,
        shuffle_files: bool = True,
        shuffle_contents: bool = True,
        random_seed: int = 0,
        device: str = "cpu",
    ):
        """Args:
        root_directory (str): The directory containing the data files
        batch_size (int): The number of examples to load in to each batch
        find_files (bool): Whether to search for files in the root_directory
        shuffle_files (bool): Whether to shuffle the filepaths
        shuffle_contents (bool): Whether to shuffle the contents of each file
        random_seed (int): The random seed for shuffling
        device (str): The device to load the data onto
        """

        self.batch_size = batch_size
        self.shuffle_contents = shuffle_contents
        self.rng = default_rng(random_seed)
        self.device = device

        self.root_directory = root_directory
        if find_files:
            filepaths = gather_files(self.root_directory)
            if shuffle_files:
                self.rng.shuffle(filepaths)
            self.relative_filepaths = [
                os.path.relpath(f, self.root_directory) for f in filepaths
            ]

        self.first_iter = True
        self._reset_state_variables()

    def __iter__(self):
        if self.relative_filepaths is None:
            raise ValueError(
                "No filepaths have been set for the DataLoader, load a previous state or set new filepaths"
            )
        elif self.batch_size is None:
            raise ValueError(
                "No batch_size has been set for the DataLoader, set a batch_size"
            )

        self.n_files = len(self.relative_filepaths)

        if not self.first_iter:
            self._reset_state_variables()
        self.first_iter = False

        self.this_file_batch_counter = 0
        self.this_file_example_counter = 0
        self.hanging_batch = None
        self._load_next_buffer()

        return self

    def _reset_state_variables(self):
        self.file_idx = 0
        self.batch_counter = 0
        self.example_counter = 0

    def __next__(self):
        if self.buffer_idx >= self.buffer_size:
            self.file_idx += 1
            self._load_next_buffer()

        if self.example_buffer is None:
            if self.hanging_batch is not None:
                batch = self.hanging_batch
                self.hanging_batch = None
                self._update_counters(len(batch), 1)
                return batch, self.batch_counter, self.file_idx
            self.file_idx = None  # Reset file_idx to None
            self.this_file_example_counter = 0
            raise StopIteration

        batch = self.example_buffer[self.buffer_idx]
        self._update_counters(len(batch), 1)
        return batch, self.batch_counter, self.file_idx

    def _update_counters(self, n_examples: int, n_batches: int):
        self.example_counter += n_examples
        self.this_file_example_counter += n_examples
        self.batch_counter += n_batches
        self.this_file_batch_counter += n_batches
        self.buffer_idx += 1

    def _load_next_buffer(self):
        def _get_next_file_data() -> torch.Tensor | None:
            file_path = os.path.join(
                self.root_directory, self.relative_filepaths[self.file_idx]
            )
            data = torch.load(file_path, map_location=self.device)
            if isinstance(data, list):
                data = torch.cat(data, dim=0)
            if self.shuffle_contents:
                data[torch.randperm(data.size(0))]

            self.this_file_batch_counter = 0
            self.this_file_example_counter = 0
            return data

        if self.file_idx >= self.n_files:
            data = None  # Don't StopIteration yet, let the hanging batch be returned
        else:
            data = _get_next_file_data()
            print(
                f"Loading file number {self.file_idx+1}/{self.n_files} (idx {self.file_idx}) containing {len(data)} examples | Done {self.example_counter} examples in {self.batch_counter} batches this session"
            )

        self.example_buffer, self.hanging_batch = self._prepare_batches(
            data, self.batch_size, self.hanging_batch
        )
        self.buffer_size = (
            len(self.example_buffer) if self.example_buffer is not None else 0
        )
        self.buffer_idx = 0

    @staticmethod
    def _prepare_batches(
        data, batch_size: int, hanging_batch: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[None, None]:
        if hanging_batch is not None:
            if data is None:
                return (
                    None,
                    None,
                )  # No more data to load, return None to force StopIteration
            else:
                data = torch.cat([hanging_batch, data], dim=0)
        n_batches = len(data) // batch_size

        # Extract hanging batch
        hanging_batch = data[n_batches * batch_size :]
        if len(hanging_batch) == 0:
            hanging_batch = None

        # Rearrange remaining data into batches
        data = data[: n_batches * batch_size]
        data = rearrange(data, "(n b) s -> n b s", b=batch_size)
        if len(data) == 0:
            data = None

        return data, hanging_batch

    def state_dict(self) -> dict:
        """Return a dictionary containing the state variables of the DataLoader. Saves from the beginning of the current file."""
        if self.this_file_example_counter > 0:
            print(
                f"Warning: state_dict() called part way through a file ({self.this_file_batch_counter} batches), saving from the beginning of the file."
            )
        state = {
            "relative_filepaths": self.relative_filepaths,
            "batch_size": self.batch_size,
            "file_idx": self.file_idx,
            "shuffle_contents": self.shuffle_contents,
            "batch_counter": self.batch_counter - self.this_file_batch_counter,
            "example_counter": self.example_counter - self.this_file_example_counter,
        }
        return state

    def load_state_dict(self, state: dict):
        """Load the state variables of the DataLoader from a dictionary."""
        self.relative_filepaths = state["relative_filepaths"]
        self.batch_size = state["batch_size"]
        self.file_idx = state["file_idx"]
        self.shuffle_contents = state["shuffle_contents"]
        self.batch_counter = state["batch_counter"]
        self.example_counter = state["example_counter"]

        # Validate the filepaths
        for f in self.relative_filepaths:
            if not os.path.exists(os.path.join(self.root_directory, f)):
                raise FileNotFoundError(f"File {f} does not exist")

    def get_relative_filepaths(self) -> list[str]:
        return self.relative_filepaths

    def get_current_file_relpath(self) -> str:
        return self.relative_filepaths[self.file_idx]
