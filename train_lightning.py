import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import torch
import huggingface
import standalone_hyenadna
from torch.utils.data import DataLoader
from utils.dataset_splitter import DatasetSplitter
from src.dataloaders.datasets.pair_alignment_dataset import SequencePairSimilarityDataset
import os
import sys

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
    
    def forward(self, seq1, seq2):
        return self.model(seq1, seq2)
    
    def training_step(self, batch, batch_idx):
        seq1, seq2, target = batch
        print(f"pair score: {target}")
        output = self(seq1, seq2)
        loss = self.loss_fn(output, target.float())
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        seq1, seq2, target = batch
        output = self(seq1, seq2)
        loss = self.loss_fn(output, target.float())
        probs = torch.sigmoid(output)
        pred = (probs > 0.5).long()
        correct = pred.eq(target.view_as(pred)).sum().item()
        accuracy = correct / len(target)
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.log("accuracy", accuracy, on_step=False, on_epoch=True)
        
        return {"val_loss": loss, "accuracy": accuracy}
        
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
        splitter = DatasetSplitter(0.7, self.data_path)
        train_data, test_data = splitter.data
        
        self.ds_train = SequencePairSimilarityDataset(
            train_data,
            max_length=self.max_length,
            tokenizer=self.tokenizer,
            use_padding=self.use_padding,
            add_eos=self.add_eos,
        )
        
        self.ds_test = SequencePairSimilarityDataset(
            test_data,
            max_length=self.max_length,
            tokenizer=self.tokenizer,
            use_padding=self.use_padding,
            add_eos=self.add_eos,
        )
    
    def train_dataloader(self):
        return DataLoader(self.ds_train, batch_size=self.batch_size, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.ds_test, batch_size=self.batch_size, shuffle=False)

if __name__ == "__main__":
    job_id = sys.argv[1]
    batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else 128
    learning_rate = float(sys.argv[3]) if len(sys.argv) > 3 else 6e-4
    weight_decay = float(sys.argv[4]) if len(sys.argv) > 4 else 0.1
    num_epochs = 100
    
    pretrained_model_name = 'hyenadna-small-32k-seqlen'
    max_length = 13370
    use_padding = 'max_length'
    add_eos = False
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    loss_fn = torch.nn.BCEWithLogitsLoss()
    backbone_cfg = None 
    
    module = HyenaDNAModule(
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
        data_path='/vol/csedu-nobackup/project/spadronalcala/pair_alignment/galGal6_1024_13370',
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_length=max_length,
        use_padding=use_padding,
        add_eos=add_eos,
    )
    
    logger = TensorBoardLogger("lightning_logs", name=f"version_{job_id}")
    print(logger.save_dir)
    
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        logger=logger,
        accelerator='gpu',
        devices=-1,
        accumulate_grad_batches = 5,
        precision=16,
        strategy='ddp'
    )
    
    trainer.fit(module, datamodule=data_module)
