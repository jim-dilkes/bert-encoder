from threading import Thread, Event
import queue
import time
import torch
from numpy.random import default_rng
import os

from .data_ops import gather_files


class AsynchronousBatchLoader:
    def __init__(
        self,
        root_directory: str,
        batch_size=None,
        max_queue_size=10,
        shuffle_files: bool = True,
        shuffle_contents: bool = True,
        dtype=torch.long,
        device="cpu",
        random_seed: int = 0,
        debug=False,
    ):

        ## Data loader properties
        self.root_directory = root_directory
        self.batch_size = batch_size
        self.max_queue_size = max_queue_size
        self.shuffle_files = shuffle_files
        self.shuffle_contents = shuffle_contents
        self.device = device
        self.debug = debug
        self.dtype = dtype

        ## Data loader state
        self.queue = queue.Queue()
        self.stop_loading = False
        self.dynamic_buffer = None
        self.rng = default_rng(random_seed)
        self.initial_file_idx = 0
        self.next_file_idx = 0
        self.flag_state_loaded = False
        self.iter_started = False

        ## Threading
        self.continue_load = Event()
        self.continue_load.set()  # Load allowed to run at the start
        self.load_not_running = Event()
        self.load_not_running.set()  # Load is not running at the start

        ## Files
        filepaths = gather_files(self.root_directory, file_extension=".pt")
        if shuffle_files:
            self.rng.shuffle(filepaths)
        self.relative_filepaths = [
            os.path.relpath(f, self.root_directory) for f in filepaths
        ]
        self.total_files = len(self.relative_filepaths)

        ## Counters
        self.n_returned_examples_all = 0
        self.n_returned_batches_all = 0
        self.n_returned_examples_iter = 0
        self.n_returned_batches_iter = 0
        self.n_batched_examples_iter = 0
        self.n_loaded_batches_iter = 0

    def _load_file(self, file_idx: int):
        filepath = os.path.join(self.root_directory, self.relative_filepaths[file_idx])
        print(
            f"Loading file {file_idx+1} / {self.total_files} (idx {file_idx}) | Queue: {self.queue.qsize()}, Buffer: {self.dynamic_buffer.size(0)} | Filepath: {filepath}"
        )
        data = torch.load(
            filepath,
            map_location="cpu",
        )
        if isinstance(data, list):
            data = torch.cat(data, dim=0)
        if self.shuffle_contents:
            data = data[torch.randperm(data.size(0))]
            if self.debug:
                print(f"Loaded file idx {file_idx}")
        return data

    def start(self):
        def _pause_control(continue_load: Event, load_not_running: Event):

            if not continue_load.is_set():  # set means continue as normal
                if self.debug:
                    print("Pausing data load")
                load_not_running.set()
                continue_load.wait()
                if self.debug:
                    print("Resuming data load")
                load_not_running.clear()

        def _load(continue_load: Event, load_not_running: Event):
            load_not_running.clear()

            for current_file_idx in range(self.initial_file_idx, self.total_files):
                self.next_file_idx = current_file_idx  # Next file for the state dict
                _pause_control(continue_load, load_not_running)
                while self.queue.qsize() >= self.max_queue_size:
                    _pause_control(continue_load, load_not_running)
                    time.sleep(0.1)  # Wait until the queue shrinks

                data = self._load_file(current_file_idx)
                self.dynamic_buffer = torch.cat((self.dynamic_buffer, data), dim=0)

                while len(self.dynamic_buffer) >= self.batch_size:
                    batch = self.dynamic_buffer[: self.batch_size]
                    self.dynamic_buffer = self.dynamic_buffer[self.batch_size :]

                    self.queue.put(batch.to(self.device, non_blocking=True))
                    self.n_batched_examples_iter += self.batch_size
                    self.n_loaded_batches_iter += 1

                if self.debug:
                    print(
                        f"File {current_file_idx+1} of {self.total_files} (idx {current_file_idx}) finished loading and batching, loader state: {self.n_batched_examples_iter} total examples batched, {self.dynamic_buffer.size(0)} examples currently in buffer | {self.n_loaded_batches_iter} batches loaded this epoch, {self.queue.qsize()} batches currently in queue"
                    )

            load_not_running.set()
            self.queue.put(None)
            self.dynamic_buffer = None

        # The index of the file that is currently being served by __next__
        self.thread = Thread(
            target=_load, args=(self.continue_load, self.load_not_running)
        )
        self.thread.start()

    def stop_and_reset_iteration(self):
        self.iter_started = False
        self.n_returned_examples_iter = 0
        self.n_returned_batches_iter = 0
        self.n_batched_examples_iter = 0
        self.n_loaded_batches_iter = 0
        self.initial_file_idx = 0
        self.next_file_idx = 0
        self.dynamic_buffer = None
        self.queue = queue.Queue()
        raise StopIteration

    def __iter__(self):

        if self.flag_state_loaded:
            self.start()
            self.flag_state_loaded = False
        else:
            if self.batch_size is None:
                raise ValueError(
                    "Batch size must be set or loaded before starting the iterator"
                )
            if self.dynamic_buffer is None:
                self.dynamic_buffer = torch.Tensor().to(self.dtype)

            # Loaded data counters
            self.n_batched_examples_iter = 0
            self.n_loaded_batches_iter = 0
            self.next_file_idx = 0

            self.start()

            # Returned data counters
            self.n_returned_examples_iter = 0
            self.n_returned_batches_iter = 0

        self.iter_started = True
        return self

    def _increase_counters(self, n_examples, n_batches):
        self.n_returned_examples_all += n_examples
        self.n_returned_batches_all += n_batches
        self.n_returned_examples_iter += n_examples
        self.n_returned_batches_iter += n_batches

    def __next__(self):
        while self.thread.is_alive() or not self.queue.empty():
            if not self.queue.empty():
                next_item = self.queue.get()
                if next_item is None:  # Check for the loading finished signal
                    self.stop_and_reset_iteration()
                self._increase_counters(self.batch_size, 1)
                next_item = next_item.to(
                    self.device
                )  # Pass the preloaded, memory pinned batch to the device
                return (next_item, self.batch_size)
            else:
                print("WARNING: Queue is empty, loader thread still running")
                time.sleep(0.01)
        self.stop_and_reset_iteration()

    ### State saving and loading
    def state_dict(self):
        # Set loading to pause and wait for the current loading iteration to finish
        self.continue_load.clear()
        while not self.load_not_running.is_set():
            time.sleep(0.001)

        # Record the current queue contents
        with self.queue.mutex:
            queue_contents = list(self.queue.queue)

        state_dict = {
            "relative_filepaths": self.relative_filepaths,
            "batch_size": self.batch_size,
            "buffer_size": self.max_queue_size,
            "shuffle_files": self.shuffle_files,
            "shuffle_contents": self.shuffle_contents,
            "rng_state": self.rng.bit_generator.state,
            "next_file_idx": self.next_file_idx,
            "buffer_contents": self.dynamic_buffer,
            "queue_contents": queue_contents,
            "n_batched_examples_iter": self.n_batched_examples_iter,
            "n_loaded_batches_iter": self.n_loaded_batches_iter,
            "n_returned_examples_iter": self.n_returned_examples_iter,
            "n_returned_batches_iter": self.n_returned_batches_iter,
            "n_returned_examples_all": self.n_returned_examples_all,
            "n_returned_batches_all": self.n_returned_batches_all,
        }

        self.continue_load.set()
        return state_dict

    def load_state_dict(self, state_dict):
        if self.iter_started:
            raise RuntimeError(
                "Cannot load state after the iterator has been started. Please create a new instance of the loader."
            )
        self.flag_state_loaded = True  # Do not overwrite state in __iter__

        ## Data loader properties
        self.relative_filepaths = state_dict["relative_filepaths"]
        # Validate the filepaths
        for f in self.relative_filepaths:
            if not os.path.exists(os.path.join(self.root_directory, f)):
                raise FileNotFoundError(f"File {f} does not exist")
        self.batch_size = state_dict["batch_size"]
        self.max_queue_size = state_dict["buffer_size"]
        self.shuffle_files = state_dict["shuffle_files"]
        self.shuffle_contents = state_dict["shuffle_contents"]

        ## Data loader state
        self.rng = default_rng()
        self.rng.bit_generator.state = state_dict["rng_state"]
        self.initial_file_idx = state_dict[
            "next_file_idx"
        ]  # First file to load in the iterator
        self.dynamic_buffer = state_dict["buffer_contents"]
        if self.dynamic_buffer is None:
            self.dynamic_buffer = torch.Tensor().to(self.dtype)
        self.queue = queue.Queue()
        for item in state_dict["queue_contents"]:
            self.queue.put(item)

        ## Counters
        self.n_batched_examples_iter = state_dict["n_batched_examples_iter"]
        self.n_loaded_batches_iter = state_dict["n_loaded_batches_iter"]
        self.n_returned_examples_iter = state_dict["n_returned_examples_iter"]
        self.n_returned_batches_iter = state_dict["n_returned_batches_iter"]
        self.n_returned_examples_all = state_dict["n_returned_examples_all"]
        self.n_returned_batches_all = state_dict["n_returned_batches_all"]

    @classmethod
    def from_state_dict(cls, state_dict, root_directory, device="cpu", debug=False):
        loader = cls(
            root_directory=root_directory,
            batch_size=0,
            min_queue_size=0,
            shuffle_files=False,
            shuffle_contents=False,
            device=device,
            debug=debug,
        )
        loader.load_state_dict(state_dict)
        return loader
