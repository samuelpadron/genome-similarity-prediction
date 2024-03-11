import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd

class SequencePairSimilarityDataset(torch.utils.data.Dataset):
  def __init__(
      self,
      data,
      max_length,
      use_padding=None,
      tokenizer=None,
      add_eos=False,
  ):
    self.data = data
    self.max_length = max_length
    self.use_padding = use_padding
    self.tokenizer = tokenizer
    self.add_eos = add_eos

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    seq_pair = self.data.iloc[idx]
    seq1 = seq_pair['sequence_1']
    seq2 = seq_pair['sequence_2']
    label = seq_pair['label']

    seq1 = self.tokenizer(seq1,
            add_special_tokens=False,
            padding="max_length" if self.use_padding else None,
            max_length=self.max_length,
            truncation=True,
        )  # add cls and eos token (+2)
    seq2 = self.tokenizer(seq2,
            add_special_tokens=False,
            padding="max_length" if self.use_padding else None,
            max_length=self.max_length,
            truncation=True,
        )
    # check tensors
    seq1 = seq1["input_ids"]
    seq2 = seq2["input_ids"]

    # need to handle eos here
    if self.add_eos:
      seq1.append(self.tokenizer.sep_token_id)
      seq2.append(self.tokenizer.sep_token_id)

    # convert to tensor
    seq1 = torch.LongTensor(seq1)
    seq2 = torch.LongTensor(seq2)
    
    ##FOR PYTORCH LIGHTNING:
    
    # seq_pair = torch.cat((seq1.unsqueeze(1), seq2.unsqueeze(1)), dim=1)

    # seq_pair = seq_pair.view(seq_pair.size(0), -1)

    # label = torch.tensor(label)

    return seq1, seq2, label  #label: 1 ~ true, 0 ~ false
