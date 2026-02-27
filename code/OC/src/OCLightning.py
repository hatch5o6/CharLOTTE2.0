import torch
from torch import nn
from pytorch_lightning import LightningModule, LightningDataModule
from pytorch_lightning.utilities import rank_zero_info
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from OC import Seq2Seq
from CognateDataset import CognateDataset

class OCLightning(LightningModule):
    def __init__(
        self,
        config,
        src_tokenizer,
        tgt_tokenizer
    ):
        super().__init__()
        self.config = config
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.tokenizer_asserts(self.src_tokenizer, self.tgt_tokenizer)
        self.model = Seq2Seq(
            pad_idx=src_tokenizer.to_idx(src_tokenizer.pad),
            sos_idx=src_tokenizer.to_idx(src_tokenizer.bos),
            eos_idx=src_tokenizer.to_idx(src_tokenizer.eos),
            src_vocab_size=len(self.src_tokenizer),
            tgt_vocab_size=len(self.tgt_tokenizer),
            config=self.config
        )

        self.criterion = nn.NLLLoss()
        self.save_hyperparameters(self.config)

    @staticmethod
    def tokenizer_asserts(src_tokenizer, tgt_tokenizer):
        if not all([
            src_tokenizer.bos == tgt_tokenizer.bos,
            src_tokenizer.eos == tgt_tokenizer.eos,
            src_tokenizer.pad == tgt_tokenizer.pad,
            src_tokenizer.unk == tgt_tokenizer.unk,

            src_tokenizer.to_idx(src_tokenizer.bos) == tgt_tokenizer.to_idx(tgt_tokenizer.bos),
            src_tokenizer.to_idx(src_tokenizer.eos) == tgt_tokenizer.to_idx(tgt_tokenizer.eos),
            src_tokenizer.to_idx(src_tokenizer.pad) == tgt_tokenizer.to_idx(tgt_tokenizer.pad),
            src_tokenizer.to_idx(src_tokenizer.unk) == tgt_tokenizer.to_idx(tgt_tokenizer.unk),

            src_tokenizer.special_toks == tgt_tokenizer.special_toks
        ]):
            raise ValueError(f"Tokenizers do not have matching special tokens and ids!")
        
    def forward(self, **inputs):
        return self.model(
            src=inputs["source_ids"], 
            src_lengths=inputs["src_lengths"],
            tgt=inputs["target_ids"],
            max_len=self.config["max_length"],
            teacher_forcing_ratio=___?#TODO
        )
    
    def training_step(self, batch, batch_idx):
        outputs = self(batch)

        #TODO calc loss
        loss = ()

        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=self.config["batch_size"]
        )

    def validation_step(self, batch, batch_idx):
        outputs = self(batch)

        # TODO print an example

        #TODO calc loss
        loss = ()

        self.log(
            "val_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=self.config["batch_size"]
        )

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=float(self.config["learning_rate"]))
    
    def generate(self, **inputs):
        return self.model.generate(**inputs)
    
class OCDataModule(LightningDataModule):
    def __init__(
        self,
        src_tokenizer,
        tgt_tokenizer,
        train_f,
        val_f,
        batch_size,
        max_length
    ):
        super().__init__()
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.train_f = train_f
        self.val_f = val_f
        self.batch_size = batch_size
        self.max_length = max_length

        self.train_dataset = None
        self.val_dataset = None
    
    def setup(self, stage=None):
        self.train_dataset = CognateDataset(self.train_f)
        self.val_dataset = CognateDataset(self.val_f)
    
    def collate_fn(self, batch):
        src_ids = []
        tgt_ids = []
        src_lens = []
        tgt_lens = []
        
        for b in batch:
            src, tgt = b

            _, tokenized_src = self.src_tokenizer.encode(src, max_len=self.max_length)
            _, tokenized_tgt = self.tgt_tokenizer.encode(tgt, max_len=self.max_length)

            src_ids.append(torch.tensor(tokenized_src))
            tgt_ids.append(torch.tensor(tokenized_tgt))
            src_lens.append(len(tokenized_src))
            tgt_lens.append(len(tokenized_tgt))
        
        # Pad sequences
        pad_id = self.src_tokenizer.to_idx(self.src_tokenizer.pad)
        assert pad_id == self.tgt_tokenizer.to_idx(self.tgt_tokenizer.pad)
        src_ids = pad_sequence(src_ids, batch_first=True, padding_value=pad_id)
        tgt_ids = pad_sequence(tgt_ids, batch_first=True, padding_value=pad_id)

        return {
            "source_ids": src_ids,
            "target_ids": tgt_ids,
            "source_lens": src_lens,
            "target_lens": tgt_lens
        }

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            num_workers=0
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=0
        )
    
if __name__ == "__main__":
    pass
    #TODO Write tests