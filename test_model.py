import pytorch_lightning as pl
import pandas as pd
import torch
import huggingface
import standalone_hyenadna
from torch.utils.data import DataLoader
from src.dataloaders.datasets.pair_alignment_dataset import SequencePairSimilarityDataset
import sys

class HyenaDNAModule(pl.LightningModule):
    def __init__(self, pretrained_model_name, backbone_cfg, loss_fn, learning_rate, weight_decay, device):
        super().__init__()
        self.save_hyperparameters()
        
        self.model = huggingface.HyenaDNAPreTrainedModel.from_pretrained(
            '/scratch/spadronalcala',
            pretrained_model_name,
            download=False,
            config=backbone_cfg,
            device=device,
        )        
        self.loss_fn = loss_fn
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
    
    def forward(self, seq1, seq2):
        return self.model(seq1, seq2)
    
    def test_step(self, batch, batch_idx):
        seq1, seq2, target = batch
        output = self(seq1, seq2)
        loss = self.loss_fn(output, target.float())
        probs = torch.sigmoid(output)
        pred = (probs > 0.5).long()
        correct = pred.eq(target.view_as(pred)).sum().item()
        accuracy = correct / len(target)
        self.log("val_loss", loss, prog_bar=True)
        self.log("accuracy", accuracy, prog_bar=True)
        
        return {"loss": loss, "accuracy": accuracy}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.learning_rate, 
            weight_decay=self.weight_decay,
        )
        
        return optimizer

def setup_new_dataset(data_path, tokenizer, batch_size, max_length, use_padding, add_eos):
    
    data_true = pd.read_csv(data_path + '_true.csv').drop(['blastz_score'], axis=1) #not needed for now
    data_true['label'] = 1
    data_false = pd.read_csv(data_path + '_false.csv')
    data_false['label'] = 0

    data = pd.concat([data_true, data_false])
    
    ds_new = SequencePairSimilarityDataset(
        data,
        max_length=max_length,
        tokenizer=tokenizer,
        use_padding=use_padding,
        add_eos=add_eos,
    )
    return DataLoader(ds_new, batch_size=batch_size, shuffle=False)

if __name__ == "__main__":
    checkpoint_path = sys.argv[1]
    new_data_path = sys.argv[2]
    batch_size = int(sys.argv[3]) if len(sys.argv) > 3 else 16
    learning_rate = float(sys.argv[4]) if len(sys.argv) > 4 else 6e-4
    weight_decay = float(sys.argv[5]) if len(sys.argv) > 5 else 0.1
    
    pretrained_model_name = 'hyenadna-small-32k-seqlen'
    max_length = 13370
    use_padding = 'max_length'
    add_eos = False
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    loss_fn = torch.nn.BCEWithLogitsLoss()
    backbone_cfg = None
    
    tokenizer = standalone_hyenadna.CharacterTokenizer(
        characters=['A', 'C', 'T', 'G', 'N'],
        model_max_length=max_length + 2,
        add_special_tokens=False,
        padding_side='left',
    )
    
    hparams = {
        'pretrained_model_name': pretrained_model_name,
        'backbone_cfg': backbone_cfg,
        'loss_fn': loss_fn,
        'learning_rate': learning_rate,
        'weight_decay': weight_decay,
        'device': device
    }
    
    print(hparams)
    
    model = HyenaDNAModule(
        pretrained_model_name,
        backbone_cfg,
        loss_fn,
        learning_rate,
        weight_decay,
        device,
    )
    
    model = model.load_from_checkpoint(checkpoint_path,
        pretrained_model_name=pretrained_model_name,
        backbone_cfg=backbone_cfg,
        loss_fn=loss_fn,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        device=device)
    
    new_dataloader = setup_new_dataset(new_data_path, tokenizer, batch_size, max_length, use_padding, add_eos)
    
    trainer = pl.Trainer(accelerator='gpu', devices=1, precision=16)
    
    trainer.test(model, dataloaders=new_dataloader)
