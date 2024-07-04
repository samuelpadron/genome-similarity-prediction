import sys
import torch
import huggingface
import standalone_hyenadna
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import seaborn as sns
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.cli import LightningCLI
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from utils.dataset_splitter import DatasetSplitter
from src.dataloaders.datasets.pair_alignment_dataset import SequencePairSimilarityDataset
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


class HyenaDNAModule(L.LightningModule):
    def __init__(self, pretrained_model_name, backbone_cfg, loss_fn, learning_rate, weight_decay, device):
        super().__init__()
        self.save_hyperparameters()
        self.model = huggingface.HyenaDNAPreTrainedModel.from_pretrained(
            path='/scratch/spadronalcala',
            model_name=pretrained_model_name,
            download=False,
            config=backbone_cfg,
            device=device,
        )    
        self.pretrained_model_name= pretrained_model_name    
        self.loss_fn = loss_fn
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.validation_predictions = []
        self.validation_targets = []
        self.validation_losses = []
        self.test_predictions = []
        self.test_targets = []

        
    # Training
      
    def forward(self, seq1, seq2):
        return self.model(seq1, seq2)
    
    
    def training_step(self, batch, batch_idx):
        seq1, seq2, target = batch
        output = self(seq1, seq2)
        loss = self.loss_fn(output, target.float())
        self.log_dict({"train_loss": loss, "step": self.current_epoch})
        
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
        self.validation_losses.append(loss.cpu())
    
    
    def on_validation_epoch_end(self):
        # make confusion matrix
        predictions = torch.cat(self.validation_predictions).numpy()
        targets = torch.cat(self.validation_targets).numpy()
        
        cm = confusion_matrix(targets, predictions)
        
        fig = plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm')
        plt.xlabel('Predicted label')
        plt.ylabel('Actual label')
        
        self.logger.experiment.add_figure("Confusion matrix", fig, self.current_epoch)

        # accumulate validation step loss and accuracy
        avg_loss = torch.stack([x for x in self.validation_losses]).mean().item()
        accuracy = accuracy_score(targets, predictions)
        
        self.log_dict({'val_loss': avg_loss, 'val_acc': accuracy, 'step': self.current_epoch})

        self.validation_predictions.clear()
        self.validation_targets.clear()
        self.validation_losses.clear() 

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
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.learning_rate, 
            weight_decay=self.weight_decay,
        )
        
        return optimizer
    

class HyenaDNADataModule(L.LightningDataModule):
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
    data_path = "/vol/csedu-nobackup/project/spadronalcala/pair_alignment/galGal6"
    max_length = 5000
    use_padding = 'max_length'
    add_eos = False
    pretrained_model_name = 'hyenadna-small-32k-seqlen'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    loss_fn = torch.nn.BCEWithLogitsLoss()
    backbone_cfg = None 
    
    
    print(f"training on: {data_path}")
    
    logger = TensorBoardLogger("lightning_logs", log_graph=True)


    cli = LightningCLI(
        model_class=HyenaDNAModule,
        datamodule_class=HyenaDNADataModule,
        run=False,
        save_config_callback=None,
        seed_everything_default=42,
        trainer_defaults={
                "max_epochs": 10,
                "callbacks": [ModelCheckpoint(monitor="val_acc", mode="max")],
                "accelerator": "gpu"
        }
    )
    cli.instantiate_trainer(
        logger=logger,
        accelerator='gpu',
        devices=-1,
        precision=16,
        strategy='ddp'
    )
    
    tokenizer = standalone_hyenadna.CharacterTokenizer(
        characters=['A', 'C', 'T', 'G', 'N'],
        model_max_length=max_length + 2,
        add_special_tokens=False,
        padding_side='left',
    )
    
    model = HyenaDNAModule(
        cli.model.pretrained_model_name,
        backbone_cfg,
        loss_fn,
        cli.model.learning_rate,
        cli.model.weight_decay,
        cli.model.device,
    )
    
    data_module = HyenaDNADataModule(
        data_path=data_path,
        tokenizer=tokenizer,
        batch_size=cli.datamodule.batch_size,
        max_length=max_length,
        use_padding=use_padding,
        add_eos=add_eos,
    )
    
    cli.trainer.fit(model, datamodule=data_module)

    cli.trainer.test(model, dataloaders=data_module.test_dataloader())
    
    logger.finalize('success')
    