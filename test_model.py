import sys
import torch
import huggingface
import standalone_hyenadna
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger
from src.dataloaders.datasets.pair_alignment_dataset import SequencePairSimilarityDataset
from sklearn.metrics import confusion_matrix

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
        self.test_predictions = []
        self.test_targets = []
    
    def forward(self, seq1, seq2):
        return self.model(seq1, seq2)
    
    def test_step(self, batch, batch_idx):
        seq1, seq2, target = batch
        output = self(seq1, seq2)
        loss = self.loss_fn(output, target.float())
        probs = torch.sigmoid(output)
        pred = (probs > 0.5).long()
        self.test_predictions.append(pred.cpu())
        self.test_targets.append(target.cpu())
        
        correct = pred.eq(target.view_as(pred)).sum().item()
        accuracy = correct / len(target)
        self.log("test_loss", loss, on_epoch=True, prog_bar=True)
        self.log("test_accuracy", accuracy, on_epoch=True, prog_bar=True)
        
        return {"test_loss": loss, "test_accuracy": accuracy}
    
    def on_test_epoch_end(self):
        if self.test_predictions and self.test_targets:
            predictions = torch.cat(self.test_predictions).numpy()
            targets = torch.cat(self.test_targets).numpy()
            
            cm = confusion_matrix(targets, predictions)
            
            fig = plt.figure(figsize=(10, 7))
            sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm')
            plt.xlabel('Predicted label')
            plt.ylabel('Actual label')
            
            self.logger.experiment.add_figure("Confusion matrix", fig, self.current_epoch)
            
            self.test_predictions.clear()
            self.test_targets.clear()
        else:
            print("No predictions to evaluate.")
        
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
    job_id = sys.argv[1]
    checkpoint_path = sys.argv[2]
    new_data_path = sys.argv[3]
    batch_size = int(sys.argv[4]) if len(sys.argv) > 4 else 8
    learning_rate = float(sys.argv[5]) if len(sys.argv) > 5 else 6e-4
    weight_decay = float(sys.argv[6]) if len(sys.argv) > 6 else 0.1
    
    pretrained_model_name = 'hyenadna-small-32k-seqlen'
    max_length = 32000
    use_padding = 'max_length'
    add_eos = False
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    loss_fn = torch.nn.BCEWithLogitsLoss()
    backbone_cfg = None
    
    print(f"Testing on {new_data_path}")
    
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
    
    logger = TensorBoardLogger("lightning_logs", name=f"version_{job_id}")
    
    trainer = pl.Trainer(accelerator='gpu', devices=1, precision=16)
    
    trainer.test(model, dataloaders=new_dataloader)
