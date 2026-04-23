import torch
from torch import nn
from lightning import LightningModule, LightningDataModule
from lightning.pytorch.utilities import rank_zero_info, rank_zero_only
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from OC.train.OC import Seq2Seq
from OC.train.CognateDataset import CognateDataset

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
            pad_idx=src_tokenizer.pad_idx(),
            sos_idx=src_tokenizer.bos_idx(),
            eos_idx=src_tokenizer.eos_idx(),
            src_vocab_size=len(self.src_tokenizer),
            tgt_vocab_size=len(self.tgt_tokenizer),
            config=self.config
        )

        self.criterion = nn.CrossEntropyLoss(ignore_index=src_tokenizer.pad_idx())
        self.save_hyperparameters(self.config)

    @staticmethod
    def tokenizer_asserts(src_tokenizer, tgt_tokenizer):
        if not all([
            src_tokenizer.bos == tgt_tokenizer.bos,
            src_tokenizer.eos == tgt_tokenizer.eos,
            src_tokenizer.pad == tgt_tokenizer.pad,
            src_tokenizer.unk == tgt_tokenizer.unk,

            src_tokenizer.bos_idx() == tgt_tokenizer.bos_idx(),
            src_tokenizer.eos_idx() == tgt_tokenizer.eos_idx(),
            src_tokenizer.pad_idx() == tgt_tokenizer.pad_idx(),
            src_tokenizer.unk_idx() == tgt_tokenizer.unk_idx(),

            src_tokenizer.special_toks == tgt_tokenizer.special_toks
        ]):
            raise ValueError(f"Tokenizers do not have matching special tokens and ids!")
        
    def forward(self, **inputs):
        return self.model(
            src=inputs["source_ids"], 
            src_lengths=inputs["source_lengths"],
            tgt=inputs["target_ids"],
            teacher_forcing_ratio=1.0,
            beam_width=self.config["oc_n_beams"],
            max_len=self.config["oc_max_length"]
        )
    
    def calc_loss(self, outputs, target_ids):
        B = outputs.size(0)
        T = outputs.size(1) - 1 # May need to be T - 1
        V = outputs.size(2)
        return self.criterion(
            outputs[:, 1:, :].reshape(B * T, V),  # (B*T, V) — raw logits
            target_ids[:, 1:].reshape(B * T)      # (B*T,)
        )
    
    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        target_ids = batch["target_ids"]

        loss = self.calc_loss(outputs, target_ids)

        # # print an example
        # if self.config["oc_log_train_samples"] != None:
        #     if batch_idx % self.config["oc_log_train_samples"] == 0:
        #         rank_zero_info(f"############## TRAIN BATCH {batch_idx} ##############")
        #         for seq_idx, seq in enumerate(outputs):
        #             if seq_idx % 10 == 0:
        #                 tgt_seq = target_ids[seq_idx]
        #                 hyp_segment = self.tgt_tokenizer.decode(seq.argmax(-1))
        #                 tgt_segment = self.tgt_tokenizer.decode(tgt_seq)
        #                 rank_zero_info(f"---- ({seq_idx}) ----")
        #                 rank_zero_info(f"HYP: `{hyp_segment}`")
        #                 rank_zero_info(f"TGT: `{tgt_segment}`")

        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=self.config["oc_batch_size"]
        )
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        target_ids = batch["target_ids"]

        # print an example
        if batch_idx % self.config["oc_log_val_samples"] == 0:
            rank_zero_info(f"############## VAL BATCH {batch_idx} ##############")
            for seq_idx, seq in enumerate(outputs):
                if seq_idx % 10 == 0:
                    tgt_seq = target_ids[seq_idx]
                    hyp_segment = self.tgt_tokenizer.decode(seq.argmax(-1))
                    tgt_segment = self.tgt_tokenizer.decode(tgt_seq)
                    rank_zero_info(f"---- ({seq_idx}) ----")
                    rank_zero_info(f"HYP: `{hyp_segment}`")
                    rank_zero_info(f"TGT: `{tgt_segment}`")

        loss = self.calc_loss(outputs, target_ids)

        self.log(
            "val_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=self.config["oc_batch_size"]
        )
        return loss
    
    def predict_step(self, batch, batch_idx):
        outputs = self.generate(**batch)

        source_segs = self.src_tokenizer.batch_decode(batch["source_ids"])
        target_segs = self.tgt_tokenizer.batch_decode(batch["target_ids"])
        prediction = self.tgt_tokenizer.batch_decode(outputs)

        assert len(source_segs) == len(target_segs) == len(prediction)
        results = list(zip(source_segs, target_segs, prediction))
        return results

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=float(self.config["oc_learning_rate"]),
            weight_decay=self.config["oc_weight_decay"]
        )

        lr_lambda = self.get_linear_schedule_with_warmup(
            num_warmup_steps=self.config["oc_warmup_steps"],
            num_training_steps=self.config["oc_max_steps"]
        )
        scheduler = LambdaLR(optimizer, lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1
            }
        }
    
    @staticmethod
    def get_linear_schedule_with_warmup(num_warmup_steps, num_training_steps):
        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return max(
                0.0,
                float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)),
            )
        return lr_lambda
    
    def generate(self, **inputs):
        inputs["target_ids"] = None
        return self(**inputs)
    
class OCDataModule(LightningDataModule):
    def __init__(
        self,
        src_tokenizer,
        tgt_tokenizer,
        train,
        val,
        batch_size,
        max_length
    ):
        super().__init__()
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.train = train
        self.val = val
        self.batch_size = batch_size
        self.max_length = max_length

        self.train_dataset = None
        self.val_dataset = None
    
    def setup(self, stage=None):
        self.train_dataset = CognateDataset(self.train)
        self.val_dataset = CognateDataset(self.val)
    
    def collate_fn(self, batch):
        src_ids = []
        tgt_ids = []
        src_lens = []
        tgt_lens = []
        
        for b in batch:
            src, tgt = b

            _, tokenized_src = self.src_tokenizer.encode(src, max_len=self.max_length, return_tensor=True)
            _, tokenized_tgt = self.tgt_tokenizer.encode(tgt, max_len=self.max_length, return_tensor=True)

            src_ids.append(tokenized_src)
            tgt_ids.append(tokenized_tgt)
            src_lens.append(len(tokenized_src))
            tgt_lens.append(len(tokenized_tgt))
        
        # Pad sequences
        pad_id = self.src_tokenizer.pad_idx()
        assert pad_id == self.tgt_tokenizer.pad_idx()
        src_ids = pad_sequence(src_ids, batch_first=True, padding_value=pad_id)
        tgt_ids = pad_sequence(tgt_ids, batch_first=True, padding_value=pad_id)

        return {
            "source_ids": src_ids,
            "target_ids": tgt_ids,
            "source_lengths": src_lens,
            "target_lengths": tgt_lens
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
    from CharTokenizer import CharTokenizer
    from train import read_yaml

    seed = 42
    torch.manual_seed(seed)

    src_tokenizer = CharTokenizer()
    src_tokenizer.build_vocab(corpus="/home/hatch5o6/nobackup/archive/data/COGNATE_TRAIN_CoNLL/es-an_ES-AN-RNN-0_RNN-213_S-0/cognate/train.es")

    tgt_tokenizer = CharTokenizer()
    tgt_tokenizer.build_vocab(corpus="/home/hatch5o6/nobackup/archive/data/COGNATE_TRAIN_CoNLL/es-an_ES-AN-RNN-0_RNN-213_S-0/cognate/train.an")

    print("SRC VOCAB SIZE:", len(src_tokenizer))
    print("TGT VOCAB SIZE:", len(tgt_tokenizer))

    # Test collate function and data module
    dm = OCDataModule(
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
        train="/home/hatch5o6/nobackup/archive/data/COGNATE_TRAIN_CoNLL/es-an_ES-AN-RNN-0_RNN-213_S-0/fastalign/word_list.es-an.NG.cognates.0.5.txt.byNED.txt",
        val="/home/hatch5o6/nobackup/archive/data/COGNATE_TRAIN_CoNLL/es-an_ES-AN-RNN-0_RNN-213_S-0/fastalign/word_list.es-an.NG.cognates.0.5.txt.byNED.txt",
        batch_size=32,
        max_length=100
    )
    dm.setup()

    b = 0
    for batch in dm.train_dataloader():
        print("################### {batch} ###################")
        print(batch)
        if b > 10:
            break
        b += 1
