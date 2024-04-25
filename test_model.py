import torch
import pandas as pd
import os
import sys
import huggingface
import standalone_hyenadna
from torch.utils.data import DataLoader
from src.dataloaders.datasets.pair_alignment_dataset import SequencePairSimilarityDataset
from torch import nn


def run_evaluation(model, output_file, device, data_loader):
    test_output = open(f"{output_file}", 'w')
    model.eval()
    loss_fn = torch.nn.BCEWithLogitsLoss()
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for seq1, seq2, target in data_loader:
            seq1, seq2, target = seq1.to(device), seq2.to(device), target.to(device)
            output = model(seq1, seq2)
            probs = torch.sigmoid(output)
            test_loss += (loss_fn(output, target.float())).item()
            pred = (probs > 0.5).long()
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(data_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(data_loader.dataset),
        100. * correct / len(data_loader.dataset)), file=test_output)

def evaluate_model(input_file, output_file):
    data = pd.read_csv(os.path.join('/vol/csedu-nobackup/project/spadronalcala/pair_alignment/', input_file))
    data['label'] = 1
    pretrained_model_name = 'hyenadna-tiny-1k-seqlen'
    batch_size = 128
    max_length = 500    #TODO: make function to use script from laptop to get max_length
    use_padding = 'max_length'
    add_eos = False
    
    tokenizer = standalone_hyenadna.CharacterTokenizer(
            characters=['A', 'C', 'G', 'T', 'N'],  # add DNA characters, N is uncertain
            model_max_length=max_length + 2,  # to account for special tokens, like EOS
            add_special_tokens=False,  # we handle special tokens elsewhere
            padding_side='left', # since HyenaDNA is causal, we pad on the left
    )
    
    ds_test = SequencePairSimilarityDataset(
        data,
        max_length = max_length,
        tokenizer=tokenizer,
        use_padding= use_padding,
        add_eos=add_eos
    )
    
    test_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False)

    # load trained model
    backbone_cfg = None
    model_path = 'base500.pth'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = huggingface.HyenaDNAPreTrainedModel.from_pretrained(
        '/scratch/spadronalcala',
        pretrained_model_name,
        download=True,
        config=backbone_cfg,
        device=device
    )
    model.load_state_dict(torch.load('base500.pth'))
    model.to(device)

    run_evaluation(model, output_file, device, test_loader)

if __name__ == "__main__":
    csv_file = sys.argv[1]
    output_file = sys.argv[2]
    evaluate_model(csv_file, output_file)