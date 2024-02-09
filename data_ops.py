import os
import shutil
import torch
from einops import rearrange


def mask_tokens(input_batch: torch.Tensor, mask_id:int, pad_id:int, mask_prob:float=0.1,
                vocab_low_high:tuple[int,int]=(5,15000)) -> tuple[torch.Tensor, torch.Tensor]:
    """Mask random tokens in a batch of input data."""

    assert mask_prob <= 1.0 and mask_prob >= 0
    use_device = input_batch.device

    ## Select tokens for masking
    mask_probabilities = torch.full(input_batch.shape, mask_prob, device=use_device)
    padding_mask = input_batch.eq(pad_id)
    mask_probabilities.masked_fill_(padding_mask,0)
    masked_tokens_bool = torch.bernoulli(mask_probabilities).bool()
    
    
    ## Replace selected tokens
    masked_input_batch = input_batch.clone()

    # 0.8 replace masked input tokens with mask_id
    replaced_tokens_bool = torch.bernoulli(torch.full(input_batch.shape, 0.8, device=use_device)).bool() & masked_tokens_bool
    masked_input_batch.masked_fill_(replaced_tokens_bool, mask_id)

    # 0.1 replace masked input tokens with random word
    random_tokens_bool = torch.bernoulli(torch.full(input_batch.shape, 0.5, device=use_device)).bool() & masked_tokens_bool & ~replaced_tokens_bool
    random_words = torch.randint(vocab_low_high[0],vocab_low_high[1], input_batch.shape, dtype=torch.long, device=use_device)
    masked_input_batch[random_tokens_bool] = random_words[random_tokens_bool]

    return masked_input_batch, masked_tokens_bool


def reset_directory(directory:str):
    """Delete and recreate a directory"""
    if os.path.exists(directory):
        shutil.rmtree(directory)  # Delete the directory and its contents
    os.makedirs(directory)

def gather_files(directory:str, file_extension:str=None) -> list[str]:
    """Gather a list of filepaths from a directory and its subdirectories. Optionally filter by file_extension."""
    file_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file_extension is None or file[-len(file_extension):] == file_extension:
                file_list.append(os.path.join(root, file)) 
    return file_list


class DataLoader():
    """
    DataLoader for loading and iterating through batches of data from a list of filepaths.
    - Loads data from filepaths in order, splitting each file into batches of size batch_size
    - Once a file is exhausted, it loads the next file in the list
    - If examples dont divide evenly into batches, the remaining examples are held in a hanging batch
    - The hanging batch is prepended to the next file's data
    - If there is not enough data to fill a final batch, the final hanging batch is ignored to avoid a partial batch
    """
    def __init__(self, filepaths:list[str], batch_size:int, shuffle_contents:bool=True, device:str='cpu'):
        self.filepaths = filepaths
        self.batch_size = batch_size
        self.n_files = len(filepaths)

        self.shuffle_contents = shuffle_contents

        self.device = device

    def __iter__(self):
        self.batch_counter = 0
        self.example_counter = 0
        
        self.file_idx = 0
        self.hanging_batch = None
        self._load_next_buffer()


        return self
    
    def __next__(self):
        if self.buffer_idx >= self.buffer_size:
            self._load_next_buffer()
            
        if self.example_buffer is None:
            if self.hanging_batch is not None:
                final_batch = self.hanging_batch
                self.hanging_batch = None
                return final_batch
            raise StopIteration
        
        batch = self.example_buffer[self.buffer_idx]
        self.buffer_idx += 1
        self.batch_counter += 1
        self.example_counter += self.batch_size
        return batch

    def _load_next_buffer(self)->torch.Tensor | None:
        def _get_next_file_data():
            if self.file_idx >= self.n_files:
                return None
            else:
                file_path = self.filepaths[self.file_idx]
                self.file_idx += 1
                data = torch.load(file_path, map_location=self.device)
                if isinstance(data, list):
                    data = torch.cat(data, dim=0)
                if self.shuffle_contents:
                    data[torch.randperm(data.size(0))]
                return data
        
        print(f'Loading file {self.file_idx}/{self.n_files} | Done {self.example_counter} examples in {self.batch_counter} batches', end='\r')

        data = _get_next_file_data()
        self.example_buffer, self.hanging_batch = self._prepare_batches(data, self.batch_size, self.hanging_batch)
        self.buffer_size = len(self.example_buffer)
        self.buffer_idx = 0
        
    @staticmethod
    def _prepare_batches(data, batch_size:int, hanging_batch:torch.Tensor=None) -> tuple[torch.Tensor, torch.Tensor] | tuple [None, None]:
        if hanging_batch is not None:
            if data is None:
                return None, None # No more data to load, return None to force StopIteration
            else:
                data = torch.cat([hanging_batch, data], dim=0)
        n_batches = len(data) // batch_size
        hanging_batch = data[n_batches*batch_size:]
        if len(hanging_batch) == 0:
            hanging_batch = None
        data = data[:n_batches*batch_size]
        data = rearrange(data, '(n b) s -> n b s', b=batch_size)
        return data, hanging_batch