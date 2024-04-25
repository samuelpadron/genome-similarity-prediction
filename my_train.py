import sys
import json
import os
import subprocess
import torch
import huggingface
import standalone_hyenadna
from torch.utils.data import DataLoader
from dataset_splitter import DatasetSplitter
from src.dataloaders.datasets.pair_alignment_dataset import SequencePairSimilarityDataset

def train(model, device, train_loader, optimizer, epoch, loss_fn, enable_print, job_id, log_interval=10):
        """Training loop."""
        model.train()
        train_output = open(f"train_{job_id}.txt", 'a') if enable_print else None #otherwise can't see in stdout for some reason
        
        for batch_idx, (seq1, seq2, target) in enumerate(train_loader):
            seq1, seq2, target = seq1.to(device), seq2.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(seq1, seq2)
            loss = loss_fn(output, target) #target has shape [batch_size]
            loss /= len(seq1)  # avg the loss over the batc
            loss.backward()
            optimizer.step()
            if batch_idx % log_interval == 0:
                if enable_print:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(seq1), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), loss.item()), file=train_output)
                else:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(seq1), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), loss.item()))

def test(model, device, test_loader, loss_fn, enable_print, job_id):
    """Test loop."""
    model.eval()
    test_loss = 0
    correct = 0
    counter = 0
    
    with torch.no_grad():
        for seq1, seq2, target in test_loader:
            seq1, seq2, target = seq1.to(device), seq2.to(device), target.to(device)
            output = model(seq1, seq2)
            probs = torch.sigmoid(output)
            test_loss += (loss_fn(output, target.float())).item()
            pred = (probs > 0.5).long()
            correct += pred.eq(target.view_as(pred)).sum().item()

            # if enable_print and counter % 100 == 0:
            #     print(f"model output: {output}", file=test_output)
            #     print(f"probabilities: {probs}", file=test_output)
            #     print(f"target: {target}, prediction: {pred}", file=test_output)
                
            counter += 1
            
    test_loss /= len(test_loader.dataset)

    with open(f"test_{job_id}.txt", 'a') as test_output:
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)), file=test_output)
    
def run_train(job_id, batch_size, learning_rate, weight_decay):
    # experiment settings:
    num_epochs = 100  # ~100 seems fine
    max_length = 500  # max len of sequence of dataset (of what you want) ~ should experiment with this
    use_padding = 'max_length'
    
    print(f"num_epochs: {num_epochs}")
    print(f"batch size: {batch_size}")
    print(f"learning rate: {learning_rate}")
    print(f"weight_decay: {weight_decay}")
    
    #rc_aug = True  # reverse complement augmentation
    add_eos = False  # add end of sentence token

    # for fine-tuning, only the 'tiny' model can fit on colab
    pretrained_model_name = 'hyenadna-tiny-1k-seqlen'  # use None if training from scratch

    # you can override with your own backbone config here if you want,
    # otherwise we'll load the HF one by default
    backbone_cfg = None

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device:", device)

    # use pretrained Huggingface wrapper
    model = huggingface.HyenaDNAPreTrainedModel.from_pretrained(
        '/scratch/spadronalcala',
        pretrained_model_name,
        download=True,
        config=backbone_cfg,
        device=device
    )

    # create tokenizer
    tokenizer = standalone_hyenadna.CharacterTokenizer(
        characters=['A', 'C', 'G', 'T', 'N'],  # add DNA characters, N is uncertain
        model_max_length=max_length + 2,  # to account for special tokens, like EOS
        add_special_tokens=False,  # we handle special tokens elsewhere
        padding_side='left', # since HyenaDNA is causal, we pad on the left
    )

    dataset = DatasetSplitter(0.7, '/vol/csedu-nobackup/project/spadronalcala/pair_alignment/500')
    train_data, test_data = dataset.data

    ds_train = SequencePairSimilarityDataset(
        train_data,
        max_length = max_length,
        tokenizer=tokenizer,
        use_padding= use_padding,
        add_eos=add_eos
    )

    ds_test = SequencePairSimilarityDataset(
        test_data,
        max_length = max_length,
        tokenizer=tokenizer,
        use_padding= use_padding,
        add_eos=add_eos
    )

    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False)

    # loss function
    loss_fn = torch.nn.BCEWithLogitsLoss()

    # create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    model.to(device)

    for epoch in range(num_epochs):
        train(model, device, train_loader, optimizer, epoch, loss_fn, True, job_id)
        test(model, device, test_loader, loss_fn, True, job_id)
        optimizer.step()
        
    #save model 
    # save_path = os.path.join(os.getcwd(), f"model_{job_id}.pth")
    # torch.save(model.state_dict(), save_path)
    # print("Model trained and saved successfully")

if __name__ == "__main__":
    job_id = sys.argv[1]
    batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else 128
    learning_rate = float(sys.argv[3]) if len(sys.argv) > 3 else 6e-4
    weight_decay = float(sys.argv[4]) if len(sys.argv) > 4 else 0.1
    
    run_train(job_id, batch_size, learning_rate, weight_decay)