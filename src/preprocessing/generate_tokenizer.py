import string
import argparse
from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    trainers,
    Tokenizer
)
from datasets import load_dataset

def main(vocab_size:int, length:int):
  ## Define characters
  chars_punctuation = string.punctuation
  chars_whitespace = string.whitespace
  chars_alphanumeric = string.ascii_letters[:26] + string.digits
  all_chars = list(chars_punctuation + chars_whitespace + chars_alphanumeric)

  tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
  tokenizer.normalizer = normalizers.Sequence([
      normalizers.NFKD(),
      normalizers.Replace("\n", " "),
      normalizers.Lowercase(),
      normalizers.StripAccents(),
      normalizers.Strip()
      ])
  tokenizer.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.Whitespace(), pre_tokenizers.Punctuation()])
  tokenizer.decoder = decoders.BPEDecoder(suffix="</w>")

  tokenizer.enable_truncation(max_length=length)
  tokenizer.enable_padding(length=length, pad_id=1, pad_token='[PAD]')

  trainer = trainers.BpeTrainer(vocab_size=vocab_size,
                                special_tokens=["[UNK]", "[PAD]", "[MASK]"], 
                                show_progress=True,
                                initial_alphabet=all_chars,
                                limit_alphabet=len(all_chars),
                                min_frequency=5,
                              #   continuing_subword_prefix="##",
                                end_of_word_suffix="</w>"
                                )
  

  n_select = int(1e6)
  dataset_name = "wikipedia"
  dataset_version = "20220301.en"
  split = "train"
  print(f"Loading dataset: {dataset_name}; {dataset_version}; {split}")
  dataset = load_dataset(dataset_name, dataset_version, split=split, trust_remote_code=True)
  
  print(f"Shuffling and selecting {n_select} items")
  dataset.shuffle(seed=1)
  dataset = dataset.select(range(n_select))

  print("Training tokenizer...")
  tokenizer.train_from_iterator(dataset["text"], trainer)
  print("Saving tokenizer...")
  tokenizer.save("tokenizer.json")
  

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Train a tokenizer on a directory of text files.")
    # parser.add_argument("--files_rootdir", type=str, help="The root directory containing the text files to train the tokenizer on.")
    parser.add_argument("--n_vocab", type=int, help="The vocabulary size of the tokenizer.")
    parser.add_argument("--seq_len", type=int, help="The length of the tokenized sequences in number of tokens.")
    args = parser.parse_args()
    main(args.n_vocab, args.seq_len)