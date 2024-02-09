import string, os
from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    trainers,
    Tokenizer
)

## Define characters
chars_punctuation = string.punctuation
chars_whitespace = string.whitespace
chars_alphanumeric = string.ascii_letters[:26] + string.digits
all_chars = list(chars_punctuation + chars_whitespace + chars_alphanumeric)


## Create tokenizer
vocab_size = 15000

tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
tokenizer.normalizer = normalizers.Sequence([
    normalizers.NFKD(),
    normalizers.Lowercase(),
    normalizers.StripAccents(),
    normalizers.Strip()
    ])
tokenizer.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.Whitespace(), pre_tokenizers.Punctuation()])
tokenizer.decoder = decoders.BPEDecoder(suffix="</w>")

tokenizer.enable_truncation(max_length=128,)
tokenizer.enable_padding(max_length=128, pad_id=3, pad_token='[PAD]')

trainer = trainers.BpeTrainer(vocab_size=vocab_size,
                              special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"], 
                              show_progress=True,
                              initial_alphabet=all_chars,
                              limit_alphabet=len(all_chars),
                              min_frequency=5,
                            #   continuing_subword_prefix="##",
                              end_of_word_suffix="</w>"
                              )

## Train tokenizer
filedir = "data_sent"
files = [os.path.join(filedir, f) for f in os.listdir(filedir) if os.path.isfile(os.path.join(filedir, f))]
tokenizer.train(files, trainer)
tokenizer.save("tokenizer.json")