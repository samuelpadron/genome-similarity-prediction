import torch
import pandas as pd
import os
import sys
import huggingface
import standalone_hyenadna
from torch.utils.data import DataLoader
from src.dataloaders.datasets.pair_alignment_dataset import SequencePairSimilarityDataset
from torch import nn


def run_evaluation(model, device, data_loader, output_file):
    model.eval()
    loss_fn = nn.BCEWithLogitsLoss()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for seq1, seq2, target in data_loader:
            seq1, seq2, target = seq1.to(device), seq2.to(device), target.to(device)
            
            print(f'seq1 device: {seq1.device}, model device: {next(model.parameters()).device}')
            
            output = model(seq1, seq2)
            probs = torch.sigmoid(output)
            
            print(f'output: {output[:10]}')
            print(f'probabilities: {probs[:10]}')
            print(f'targets: {target[:10]}')
            
            test_loss += loss_fn(output, target.float()).item()
            pred = (probs > 0.5).long()
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(data_loader.dataset)

    with open(output_file, 'w') as f:
        f.write(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(data_loader.dataset)} ({100. * correct / len(data_loader.dataset):.2f}%)\n')

def load_data(input_files):
    data_true = pd.read_csv(os.path.join('/vol/csedu-nobackup/project/spadronalcala/pair_alignment/', input_files) + '_true.csv')
    data_true['label'] = 1
    data_false = pd.read_csv(os.path.join('/vol/csedu-nobackup/project/spadronalcala/pair_alignment/', input_files) + '_false.csv')
    data_false['label'] = 0
    data = pd.concat([data_true, data_false])
    
    batch_size = 16
    max_length = 13370    #TODO: make function to use script from laptop to get max_length
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
    
    return test_loader

def load_model():
    # load trained model
    backbone_cfg = None
    model_path = 'model_4464002.pth'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pretrained_model_name = 'hyenadna-small-32k-seqlen'
    
    model = huggingface.HyenaDNAPreTrainedModel.from_pretrained(
        '/scratch/spadronalcala',
        pretrained_model_name,
        download=False,
        config=backbone_cfg,
        device=device
    )
    model.load_state_dict(torch.load(model_path), strict=False)
    
    print(f"model state dict keys: {model.state_dict()}")
    
    model.to(device)
    
    return model, device

def evaluate_model(input_files, output_file):
    test_loader = load_data(input_files)
    model, device = load_model()
    run_evaluation(model, device, test_loader, output_file)

if __name__ == "__main__":
    input_files = sys.argv[1]
    output_file = sys.argv[2]
    evaluate_model(input_files, output_file)