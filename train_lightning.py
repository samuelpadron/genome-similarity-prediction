import sys
import torch
import huggingface
import standalone_hyenadna
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import seaborn as sns
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from utils.dataset_splitter import DatasetSplitter
from src.dataloaders.datasets.pair_alignment_dataset import SequencePairSimilarityDataset
from sklearn.metrics import confusion_matrix


class HyenaDNAModule(pl.LightningModule):
    def __init__(self, pretrained_model_name, backbone_cfg, loss_fn, learning_rate, weight_decay, device):
        super().__init__()
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
        self.validation_predictions = []
        self.validation_targets = []

        
    # Training
      
    def forward(self, seq1, seq2):
        return self.model(seq1, seq2)
    
    
    def training_step(self, batch, batch_idx):
        seq1, seq2, target = batch
        output = self(seq1, seq2)
        loss = self.loss_fn(output, target.float())
        self.log('train_loss', loss, on_step=False, on_epoch=True, sync_dist=True)
        
        return loss
    
    
    # Validation

    def validation_step(self, batch, batch_idx):
        seq1, seq2, target = batch
        output = self(seq1, seq2)
        loss = self.loss_fn(output, target.float())
        probs = torch.sigmoid(output)
        pred = (probs > 0.5).long()
        self.validation_predictions.append(pred.cpu())
        self.validation_targets.append(target.cpu())
        correct = pred.eq(target.view_as(pred)).sum().item()
        accuracy = correct / len(target)
        self.log("val_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val_accuracy", accuracy, on_step=False, on_epoch=True, sync_dist=True)
        
        return {"val_loss": loss, "val_accuracy": accuracy}
    
    
    def on_validation_epoch_end(self):
        predictions = torch.cat(self.validation_predictions).numpy()
        targets = torch.cat(self.validation_targets).numpy()
        
        cm = confusion_matrix(targets, predictions)
        
        fig = plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm')
        plt.xlabel('Predicted label')
        plt.ylabel('Actual label')
        
        self.logger.experiment.add_figure("Confusion matrix", fig, self.current_epoch)
        
        self.validation_predictions.clear()
        self.validation_targets.clear()


    # Testing        
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
    

class HyenaDNADataModule(pl.LightningDataModule):
    def __init__(self, data_path, tokenizer, batch_size, max_length, use_padding, add_eos):
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length
        self.use_padding = use_padding
        self.add_eos = add_eos
    
    def setup(self, stage=None):
        splitter = DatasetSplitter(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, data_path=self.data_path)
        train_data, val_data, test_data = splitter.data
        
        self.ds_train = SequencePairSimilarityDataset(
            train_data,
            max_length=self.max_length,
            tokenizer=self.tokenizer,
            use_padding=self.use_padding,
            add_eos=self.add_eos
        )
        
        self.ds_val = SequencePairSimilarityDataset(
            val_data,
            max_length=self.max_length,
            tokenizer=self.tokenizer,
            use_padding=self.use_padding,
            add_eos=self.add_eos
        )
        
        self.ds_test = SequencePairSimilarityDataset(
            test_data,
            max_length=self.max_length,
            tokenizer=self.tokenizer,
            use_padding=self.use_padding,
            add_eos=self.add_eos
        )
    
    def train_dataloader(self):
        return DataLoader(self.ds_train, batch_size=self.batch_size, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.ds_val, batch_size=self.batch_size, shuffle=True)
    
    def test_dataloader(self):
        return DataLoader(self.ds_test, batch_size=self.batch_size, shuffle=True)

if __name__ == "__main__":
    job_id = sys.argv[1]
    batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else 128
    learning_rate = float(sys.argv[3]) if len(sys.argv) > 3 else 6e-4
    weight_decay = float(sys.argv[4]) if len(sys.argv) > 4 else 0.1
    data_path = sys.argv[5]
    
    print(f"training on: {data_path}")
    
    num_epochs = 100
    max_length = 5000
    use_padding = 'max_length'
    add_eos = False
    pretrained_model_name = 'hyenadna-small-32k-seqlen'
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    loss_fn = torch.nn.BCEWithLogitsLoss()
    backbone_cfg = None 
    
    model = HyenaDNAModule(
        pretrained_model_name,
        backbone_cfg,
        loss_fn,
        learning_rate,
        weight_decay,
        device,
    )
    
    tokenizer = standalone_hyenadna.CharacterTokenizer(
        characters=['A', 'C', 'T', 'G', 'N'],
        model_max_length=max_length + 2,
        add_special_tokens=False,
        padding_side='left',
    )
    
    data_module = HyenaDNADataModule(
        data_path=data_path,
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_length=max_length,
        use_padding=use_padding,
        add_eos=add_eos,
    )
    
    logger = TensorBoardLogger("lightning_logs", name=f"version_{job_id}", log_graph=True)
    
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        logger=logger,
        accelerator='gpu',
        devices=-1,
        precision=16,
        strategy='ddp'
    )
    
    trainer.fit(model, datamodule=data_module)

    trainer.test(model, dataloaders=data_module.test_dataloader)
    
    logger.finalize()
    
