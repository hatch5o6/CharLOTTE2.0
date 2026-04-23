import torch
from lightning.pytorch import LightningModule, LightningDataModule
from lightning.pytorch.utilities import rank_zero_info
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from transformers import BartForConditionalGeneration, BartConfig

from NMT.train.ParallelDatasets import CharLOTTEParallelDataset

class BARTLightning(LightningModule):
    """
    Template model class. 
    """
    def __init__(
        self,
        config,
        tokenizer,
    ):
        """
        Define your model here. Use self.model
        """
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.model = self.get_bart_model(
            tokenizer=self.tokenizer, 
            config=self.config)

        self.save_hyperparameters(self.config)

    def get_bart_model(self, tokenizer, config):
        model_vocab_size = len(tokenizer)
        rank_zero_info(f"MODEL VOCAB SIZE: {model_vocab_size}")
        
        # TODO Set all BartConfigs :)
        rank_zero_info("\nget_bart_model() SETTING CONFIGS:")
        for k,v in config.items():
            rank_zero_info(f"{k}:`{v}`")
        rank_zero_info("\n")
        model_config = BartConfig()
        model_config.vocab_size = model_vocab_size
        model_config.pad_token_id=tokenizer.pad_token_id
        model_config.bos_token_id=tokenizer.bos_token_id
        model_config.eos_token_id=tokenizer.eos_token_id
        model_config.forced_eos_token_id=tokenizer.eos_token_id
        model_config.decoder_start_token_id=tokenizer.bos_token_id #
        
        # Model Config
        model_config.encoder_layers = config["nmt_enc_num_layers"]
        model_config.decoder_layers = config["nmt_dec_num_layers"]
        model_config.encoder_attention_heads = config["nmt_enc_att_heads"]
        model_config.decoder_attention_heads = config["nmt_dec_att_heads"]
        model_config.encoder_ffn_dim = config["nmt_enc_ffn_dim"]
        model_config.decoder_ffn_dim = config["nmt_dec_ffn_dim"]
        model_config.encoder_layerdrop = config["nmt_enc_layerdrop"]
        model_config.decoder_layerdrop = config["nmt_dec_layerdrop"]

        model_config.max_position_embeddings = config["nmt_max_position_embeddings"]
        model_config.d_model = config["nmt_d_model"]
        model_config.dropout = config["nmt_dropout"]

        model_config.activation_function = config["nmt_activation"]

        model = BartForConditionalGeneration(model_config)

        return model

    def forward(self, **inputs):
        """
        Define the forward pass here. If using a HuggingFace model,
        the loss can be computed automatically by passing 'labels' in inputs.
        """

        input_ids = inputs["source_ids"]
        labels = inputs["target_ids"]

        return self.model(input_ids=input_ids, labels=labels)

    def training_step(self, batch, batch_idx):
        """
        The batch structure is defined by the collate function from your DataModule.
        Unpack the batch, do a forward pass, and return the loss.
        """

        # Unpack batch
        input_ids = batch['source_ids']
        labels = batch['target_ids']

        # Forward pass
        outputs = self(
            source_ids=input_ids, 
            target_ids=labels
        )
        
        # Log and return loss
        loss = outputs.loss
        self.log(
            "train_loss", 
            loss, 
            on_step=True, 
            on_epoch=True,
            prog_bar=True, 
            logger=True,
            batch_size=self.config["nmt_batch_size"]
        )
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Same as train step except you should also print an example output for debugging and monitoring.
        """
        input_ids = batch['source_ids']
        labels = batch['target_ids']

        outputs = self(
            source_ids=input_ids, 
            target_ids=labels
        )

        # TODO: Print an example here
        if batch_idx % self.config["nmt_log_val_samples"] == 0:
            rank_zero_info(f"############## VAL BATCH {batch_idx} ##############")
            with torch.no_grad():
                gen_ids = self.model.generate(
                    input_ids=input_ids[:1],
                    max_new_tokens=self.config.get("nmt_max_length", 128)
                )
            src_text = self.tokenizer.decode(input_ids[0])
            ref_text = self.tokenizer.decode(labels[0].masked_fill(labels[0] == -100, self.tokenizer.pad_token_id))
            hyp_text = self.tokenizer.decode(gen_ids[0])

            rank_zero_info(f"\tSRC: `{src_text}`")
            rank_zero_info(f"\tREF: `{ref_text}`")
            rank_zero_info(f"\tHYP: `{hyp_text}`")

        loss = outputs.loss
        self.log(
            "val_loss", 
            loss, 
            on_step=True,
            on_epoch=True,
            prog_bar=True, 
            logger=True,
            batch_size=self.config["nmt_batch_size"],
            sync_dist=True
        )
        return loss
    
    def predict_step(self, batch, batch_idx):
        outputs = self.generate(**batch)
        source_segs = self.tokenizer.batch_decode(batch["source_ids"])
        target_ids = batch["target_ids"].masked_fill(batch["target_ids"] == -100, self.tokenizer.pad_token_id)
        target_segs = self.tokenizer.batch_decode(target_ids)
        prediction = self.tokenizer.batch_decode(outputs)

        assert len(source_segs) == len(target_segs) == len(prediction)
        results = list(zip(source_segs, target_segs, prediction))
        return results

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=float(self.config["nmt_learning_rate"]),
            weight_decay=self.config["nmt_weight_decay"]
        )

        lr_lambda = self.get_linear_schedule_with_warmup(
            num_warmup_steps=self.config["nmt_warmup_steps"],
            num_training_steps=self.config["nmt_max_steps"]
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
        """
        Define the generation method for inference here. You may need to 
        implement this yourself depending on the model.
        """
        # Example, wrapper for the HuggingFace generate method (for inference)
        return self.model.generate(input_ids=inputs["source_ids"], max_length=self.config["nmt_max_length"])


class BARTDataModule(LightningDataModule):
    def __init__(
        self, 
        tokenizer, 
        data:list, # list of [data folder, pl, cl, tl] tuples
        sc_model_ids:dict={}, 
        reverse:bool=False,
        mode:str="parent",
        batch_size:int=32, 
        max_length:int=128, 
        append_lang_tags:bool=False
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length
        self.sc_model_ids = sc_model_ids
        self.reverse = reverse
        self.append_lang_tags = append_lang_tags
        self.vocab_size = len(tokenizer)

        if mode not in ["parent", "child", "oc"]:
            raise ValueError(f"mode must be 'parent', 'child', or 'oc'")
        self.mode = mode

        for item in data:
            if not (isinstance(item, list) or isinstance(item, tuple)):
                raise ValueError("datasets must be a list of tuples/lists!")
            if len(item) != 4:
                raise ValueError("Each item in datasets must be a tuple/list of length 4: data folder, pl, cl, tl")
        self.data = data
        
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage=None):
        self.train_dataset = CharLOTTEParallelDataset(
            datasets=self.data,
            sc_model_ids=self.sc_model_ids,
            reverse=self.reverse,
            mode=self.mode,
            div="train"
        )

        self.val_dataset = CharLOTTEParallelDataset(
            datasets=self.data,
            sc_model_ids=self.sc_model_ids,
            reverse=self.reverse,
            mode=self.mode,
            div="val"
        )

        self.test_dataset = CharLOTTEParallelDataset(
            datasets=self.data,
            sc_model_ids=self.sc_model_ids,
            reverse=self.reverse,
            mode=self.mode,
            div="test"
        )

    def collate_fn(self, batch):
        """
        Structure of source and target:

        Multilingual:
        Source: <lang> ...data... </s>
        Target: <lang> ...data... </s>

        Bilingual:
        Source: ...data... </s>
        Target: ...data... </s>
        """
        src_ids = []
        tgt_ids = []
        src_lens = []
        tgt_lens = []
        src_lang_ids = []
        tgt_lang_ids = []

        for b in batch:
            src = b['src']
            tgt = b['tgt']
            src_lang = b['src_lang']
            tgt_lang = b['tgt_lang']

            # Tokenize source and target
            tokenized_src = self.tokenizer(src, add_special_tokens=False).input_ids
            tokenized_tgt = self.tokenizer(tgt, add_special_tokens=False).input_ids

            # Limit to max length
            seq_buffer = 2 if self.append_lang_tags else 1
            tokenized_src = tokenized_src[:self.max_length - seq_buffer] # -1 for eos, or (if appending lang tags) -2 for lang, eos
            tokenized_tgt = tokenized_tgt[:self.max_length - seq_buffer] # -1 for eos, or (if appending lang tags) -2 for lang, eos
                
            # Add EOS token
            tokenized_src = tokenized_src + [self.tokenizer.eos_token_id]
            tokenized_tgt = tokenized_tgt + [self.tokenizer.eos_token_id]

            # Add special tokens for language
            if self.append_lang_tags:
                src_lang_token = self.tokenizer.convert_tokens_to_ids(f"<{src_lang}>")
                tgt_lang_token = self.tokenizer.convert_tokens_to_ids(f"<{tgt_lang}>")

                tokenized_src = [src_lang_token] + tokenized_src
                tokenized_tgt = [tgt_lang_token] + tokenized_tgt
            else:
                src_lang_token = None
                tgt_lang_token = None

            src_ids.append(torch.tensor(tokenized_src, dtype=torch.long))
            tgt_ids.append(torch.tensor(tokenized_tgt, dtype=torch.long))
            src_lens.append(len(tokenized_src))
            tgt_lens.append(len(tokenized_tgt))
            src_lang_ids.append(src_lang_token)
            tgt_lang_ids.append(tgt_lang_token)

        # Pad sequences with self.text_tokenizer.pad_token_id
        src_ids = pad_sequence(src_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        tgt_ids = pad_sequence(tgt_ids, batch_first=True, padding_value=-100) # -100 ignores loss on padding


        # Return structure that interfaces with the Model class above:
        return {
            "source_lang_ids": torch.tensor(src_lang_ids, dtype=torch.int),
            "target_lang_ids": torch.tensor(tgt_lang_ids, dtype=torch.int),
            "source_lens": torch.tensor(src_lens, dtype=torch.int),
            "target_lens": torch.tensor(tgt_lens, dtype=torch.int),
            "source_ids": src_ids,
            "target_ids": tgt_ids
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
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=0
        )
    
# def main():
#     import os
#     from Pipeline.Pipeline.pipeline import read_config
#     config = read_config("/home/hatch5o6/CharLOTTE2.0/src/configs/test.yaml")
#     torch.manual_seed(config["seed"])

#     # tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
#     from NMT.train.NMTTokenizer import (
#         train_unigram, 
#         load_tokenizer, 
#         make_tokenizer_data
#     )
#     tokenizer_files, tokenizer_dir = make_tokenizer_data(config)
#     assert len(tokenizer_files) == 1 and ("es", "an", "en") in tokenizer_files.keys()
#     tokenizer_files = tokenizer_files[("es", "an", "en")]
#     unigram_tokenizer_path = train_unigram(files=tokenizer_files,
#                                           save=tokenizer_dir,
#                                           vocab_size=config["nmt_vocab_size"],
#                                           bos=config["nmt_bos"],
#                                           eos=config["nmt_eos"],
#                                           pad=config["nmt_pad"],
#                                           unk=config["nmt_unk"],
#                                           seed=config["seed"])
#     tokenizer = load_tokenizer(unigram_tokenizer_path)
    
#     print("Vocab size:", len(tokenizer))
#     assert len(tokenizer) == config["nmt_vocab_size"]

#     # CONTINUE FROM HERE
#     # Test collate function
#     dm = BARTDataModule(
#         tokenizer=tokenizer,
#         data=?,
#         sc_model_ids=?
#     )
#     """
#     class BARTDataModule(LightningDataModule):
#         def __init__(
#         self, 
#         tokenizer, 
#         data:list, # list of [data folder, pl, cl, tl] tuples
#         sc_model_ids:dict={}, 
#         reverse:bool=False,
#         mode:str="parent",
#         batch_size:int=32, 
#         max_length:int=128, 
#         append_lang_tags:bool=False
#     ):
#     """
#     dm.setup()

#     model = BARTLightning(config=____, tokenizer=tokenizer)

#     optim = torch.optim.Adam(model.parameters(), lr=3e-4)

#     loader = dm.train_dataloader()
#     for batch in loader:

#         print(batch['target_ids'])

#         for step in range(40):
#             loss = model.training_step(batch, 0)
#             loss.backward()
#             optim.step()
#             optim.zero_grad()
#             print(loss)

#         source_ids = batch['source_ids']
#         target_ids = batch['target_ids']

#         source_lens = batch['source_lens']
#         target_lens = batch['target_lens']

#         source_lang_ids = batch['source_lang_ids']
#         target_lang_ids = batch['target_lang_ids']

#         print(target_ids)

#         with torch.no_grad():
#             test = model.generate(
#                 input_ids=source_ids,
#                 source_lens=source_lens,
#                 target_lang_ids=target_lang_ids,
#                 num_beams=2,
#                 # max_length=10,
#             )

#             print(test)
#             break

if __name__ == "__main__":
    pass
