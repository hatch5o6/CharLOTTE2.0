import os
import shutil
import torch

from NMT.train.BARTLightning import BARTDataModule
from NMT.train.ParallelDatasets import CharLOTTEParallelDataset
from NMT.train.NMTTokenizer import make_tokenizer_data, train_unigram, load_tokenizer
import utilities
from utilities.utilities import set_vars_in_path, set_env

# test the collate_fn function mainly
#TODO test multilingual dataset

BATCH_ITEM_TO_TRUNCATE = {
    "src_lang": "an",
    "tgt_lang": "en",
    "src_path": "None",
    "tgt_path": "None",
    "src": ("Que dengún no se mueva! " * 100).strip(),
    "tgt": ("Don't let anyone move please" * 100).strip()
}

class TestBARTDataModule:
    @classmethod
    def setup_class(cls):
        set_env()

        # Bilingual xx|xx-->xx tokenizer
        cls.config_f = "src/configs/test.yaml"
        cls.config = utilities.read_data.read_config(cls.config_f)

        cls.save_dir = cls.config["save"]
        assert os.path.exists(cls.save_dir)

        cls.exp_dir = os.path.join(cls.save_dir, cls.config["experiment_name"])
        if os.path.exists(cls.exp_dir):
            print("REMOVING", cls.exp_dir)
            shutil.rmtree(cls.exp_dir)
        assert not os.path.exists(cls.exp_dir)
        print("CREATING", cls.exp_dir)
        os.mkdir(cls.exp_dir)

        os.mkdir(os.path.join(cls.exp_dir, "NMT"))

        print("\tMaking data")
        cls.tokenizer_train_data, cls.tokenizer_dir = make_tokenizer_data(cls.config)
        assert len(cls.config["data"]) == 1
        assert len(cls.tokenizer_train_data) == 1
        cls.scenario = tuple(cls.config["data"][0][1:])
        cls.data_files = cls.tokenizer_train_data[cls.scenario]

        print("\tTraining Unigram")
        cls.unigram_tokenizer_dir = train_unigram(files=cls.data_files,
                                                  save=cls.tokenizer_dir,
                                                  vocab_size=cls.config["nmt_vocab_size"],
                                                  seed=cls.config["seed"],
                                                  lang_toks=["<an>", "<en>"])
        cls.tokenizer = load_tokenizer(cls.unigram_tokenizer_dir)

        cls.max_length = 128
        cls.batch_size = 32

        cls.dm = BARTDataModule(
            tokenizer=cls.tokenizer,
            data=cls.config["data"],
            batch_size=cls.batch_size,
            max_length=cls.max_length
        )
        cls.dm.setup()

        cls.dm_with_lang_tags = BARTDataModule(
            tokenizer=cls.tokenizer,
            data=cls.config["data"],
            batch_size=cls.batch_size,
            max_length=cls.max_length,
            append_lang_tags=True
        )
        cls.dm_with_lang_tags.setup()

    def test_collate_fn_output_format(self):
        for data_loader in [self.dm.val_dataloader(), self.dm_with_lang_tags.val_dataloader()]:
            data_loader_length = len(data_loader)
            for i, result in enumerate(data_loader):
                assert isinstance(result, dict)
                gt_keys = ["source_lang_ids",
                        "target_lang_ids",
                        "source_lens",
                        "target_lens",
                        "source_ids",
                        "target_ids"]
                assert list(result.keys()) == gt_keys
                for k in gt_keys:
                    assert torch.is_tensor(result[k])
                    if i < data_loader_length - 1:
                        assert result[k].shape[0] == self.batch_size
                    else:
                        assert result[k].shape[0] <= self.batch_size
                    if k in ["source_ids", "target_ids"]:
                        assert result[k].dtype == torch.long
                        assert result[k].ndim == 2
                    else:
                        assert result[k].dtype.is_floating_point is False
                        assert result[k].ndim == 1

    def test_collate_fn_max_length_respected(self):
        for dataloader in [self.dm.val_dataloader(), 
                           self.dm.train_dataloader(), 
                           self.dm.test_dataloader()]:
            for result in dataloader:
                for src_len in result["source_lens"]:
                    assert src_len.item() <= self.max_length
                for tgt_len in result["target_lens"]:
                    assert tgt_len.item() <= self.max_length
                assert result["source_ids"].shape[1] <= self.max_length
                assert result["target_ids"].shape[1] <= self.max_length
    
    def test_collate_fn_max_length_respected_with_lang_tags(self):
        for dataloader in [self.dm_with_lang_tags.val_dataloader(),
                           self.dm_with_lang_tags.train_dataloader(),
                           self.dm_with_lang_tags.test_dataloader()]:
            for result in dataloader:
                for src_len in result["source_lens"]:
                    assert src_len.item() <= self.max_length
                for tgt_len in result["target_lens"]:
                    assert tgt_len.item() <= self.max_length
                assert result["source_ids"].shape[1] <= self.max_length
                assert result["target_ids"].shape[1] <= self.max_length

    def test_collate_fn_pads_source(self):
        for dataloader in [self.dm.val_dataloader(), self.dm_with_lang_tags.val_dataloader()]:
            for result in dataloader:
                max_seq_len = max(result["source_lens"]).item()
                for i, seq in enumerate(result["source_ids"]):
                    seq_len = result["source_lens"][i].item()
                    assert seq.ndim == 1 and seq.shape[0] == max_seq_len
                    og_seq = seq[:seq_len]
                    pads = seq[seq_len:]
                    for item in og_seq:
                        assert item.item() != self.tokenizer.pad_token_id
                    for item in pads:
                        assert item.item() == self.tokenizer.pad_token_id

    def test_collate_fn_pads_target(self):
        for dataloader in [self.dm.val_dataloader(), self.dm_with_lang_tags.val_dataloader()]:
            for result in dataloader:
                max_seq_len = max(result["target_lens"]).item()
                for i, seq in enumerate(result["target_ids"]):
                    seq_len = result["target_lens"][i].item()
                    assert seq.ndim == 1 and seq.shape[0] == max_seq_len
                    og_seq = seq[:seq_len]
                    pads = seq[seq_len:]
                    for item in og_seq:
                        assert item.item() != -100
                    for item in pads:
                        assert item.item() == -100

    def test_collate_fn_eos_appended_to_source(self):
        for dataloader in [self.dm.val_dataloader(), self.dm_with_lang_tags.val_dataloader()]:
            for result in dataloader:
                max_seq_len = max(result["source_lens"]).item()
                for i, seq in enumerate(result["source_ids"]):
                    seq_len = result["source_lens"][i].item()
                    assert seq.ndim == 1 and seq.shape[0] == max_seq_len
                    og_seq = seq[:seq_len]
                    assert og_seq[-1].item() == self.tokenizer.eos_token_id
                    
    
    def test_collate_fn_eos_appended_to_target(self):
        for dataloader in [self.dm.val_dataloader(), self.dm_with_lang_tags.val_dataloader()]:
            for result in dataloader:
                max_seq_len = max(result["target_lens"]).item()
                for i, seq in enumerate(result["target_ids"]):
                    seq_len = result["target_lens"][i].item()
                    assert seq.ndim == 1 and seq.shape[0] == max_seq_len
                    og_seq = seq[:seq_len]
                    assert og_seq[-1].item() == self.tokenizer.eos_token_id

    def test_collate_fn_truncates_source(self):
        batch = [BATCH_ITEM_TO_TRUNCATE]
        src_toks = self.tokenizer.encode(BATCH_ITEM_TO_TRUNCATE["src"], add_special_tokens=False)
        tgt_toks = self.tokenizer.encode(BATCH_ITEM_TO_TRUNCATE["tgt"], add_special_tokens=False)

        print("BATCH ITEM TO TRUNCATE src length:", len(src_toks))
        print("BATCH ITEM TO TRUNCATE tgt length:", len(tgt_toks))

        assert len(src_toks) > self.max_length

        result = self.dm.collate_fn(batch)

        assert result["source_lens"][0].item() == self.max_length
        assert len(result["source_ids"][0]) == self.max_length

        assert result["source_ids"][0][-1].item() == self.tokenizer.eos_token_id
        # self.tokenizer.get_added_vocab() --> {'<pad>': 0, '<unk>': 1, '<bos>': 2, '<eos>': 3, '<an>': 4, '<en>': 5}
        assert result["source_ids"][0][0].item() not in self.tokenizer.get_added_vocab().values()
        assert result["source_ids"][0][0].item() not in self.tokenizer.all_special_ids
        
        for i, tok_id in enumerate(result["source_ids"][0]):
            if i == len(result["source_ids"][0]) - 1:
                assert tok_id == self.tokenizer.eos_token_id
            else:
                assert tok_id == src_toks[i]

        source_sequence = self.tokenizer.decode(result["source_ids"][0])
        assert source_sequence.endswith("<eos>")
        assert not source_sequence.startswith("<an>")
        assert not source_sequence.startswith("<en>")

    
    def test_collate_fn_truncates_source_with_lang_tags(self):
        batch = [BATCH_ITEM_TO_TRUNCATE]
        src_toks = self.tokenizer.encode(BATCH_ITEM_TO_TRUNCATE["src"], add_special_tokens=False)
        tgt_toks = self.tokenizer.encode(BATCH_ITEM_TO_TRUNCATE["tgt"], add_special_tokens=False)

        print("BATCH ITEM TO TRUNCATE src length:", len(src_toks))
        print("BATCH ITEM TO TRUNCATE tgt length:", len(tgt_toks))

        assert len(src_toks) > self.max_length

        result = self.dm_with_lang_tags.collate_fn(batch)

        assert result["source_lens"][0].item() == self.max_length
        assert len(result["source_ids"][0]) == self.max_length

        assert result["source_ids"][0][-1].item() == self.tokenizer.eos_token_id
        # self.tokenizer.get_added_vocab() --> {'<pad>': 0, '<unk>': 1, '<bos>': 2, '<eos>': 3, '<an>': 4, '<en>': 5}
        assert result["source_ids"][0][0].item() == self.tokenizer.convert_tokens_to_ids("<an>")

        for i, tok_id in enumerate(result["source_ids"][0]):
            if i == 0:
                assert tok_id == self.tokenizer.convert_tokens_to_ids("<an>")
            elif i == len(result["source_ids"][0]) - 1:
                assert tok_id == self.tokenizer.eos_token_id
            else:
                assert tok_id == src_toks[i - 1]

        source_sequence = self.tokenizer.decode(result["source_ids"][0])
        assert source_sequence.endswith("<eos>")
        assert source_sequence.startswith("<an>")

    def test_collate_fn_truncates_target(self):
        batch = [BATCH_ITEM_TO_TRUNCATE]
        src_toks = self.tokenizer.encode(BATCH_ITEM_TO_TRUNCATE["src"], add_special_tokens=False)
        tgt_toks = self.tokenizer.encode(BATCH_ITEM_TO_TRUNCATE["tgt"], add_special_tokens=False)

        print("BATCH ITEM TO TRUNCATE src length:", len(src_toks))
        print("BATCH ITEM TO TRUNCATE tgt length:", len(tgt_toks))

        assert len(tgt_toks) > self.max_length

        result = self.dm.collate_fn(batch)

        assert result["target_lens"][0].item() == self.max_length
        assert len(result["target_ids"][0]) == self.max_length

        assert result["target_ids"][0][-1].item() == self.tokenizer.eos_token_id
        # self.tokenizer.get_added_vocab() --> {'<pad>': 0, '<unk>': 1, '<bos>': 2, '<eos>': 3, '<an>': 4, '<en>': 5}
        assert result["target_ids"][0][0].item() not in self.tokenizer.get_added_vocab().values()
        assert result["target_ids"][0][0].item() not in self.tokenizer.all_special_ids
        
        for i, tok_id in enumerate(result["target_ids"][0]):
            if i == len(result["target_ids"][0]) - 1:
                assert tok_id == self.tokenizer.eos_token_id
            else:
                assert tok_id == tgt_toks[i]

        target_sequence = self.tokenizer.decode(result["target_ids"][0])
        assert target_sequence.endswith("<eos>")
        assert not target_sequence.startswith("<an>")
        assert not target_sequence.startswith("<en>")
    
    def test_collate_fn_truncates_target_with_lang_tags(self):
        batch = [BATCH_ITEM_TO_TRUNCATE]
        src_toks = self.tokenizer.encode(BATCH_ITEM_TO_TRUNCATE["src"], add_special_tokens=False)
        tgt_toks = self.tokenizer.encode(BATCH_ITEM_TO_TRUNCATE["tgt"], add_special_tokens=False)

        print("BATCH ITEM TO TRUNCATE src length:", len(src_toks))
        print("BATCH ITEM TO TRUNCATE tgt length:", len(tgt_toks))

        assert len(tgt_toks) > self.max_length

        result = self.dm_with_lang_tags.collate_fn(batch)

        assert result["target_lens"][0].item() == self.max_length
        assert len(result["target_ids"][0]) == self.max_length

        assert result["target_ids"][0][-1].item() == self.tokenizer.eos_token_id
        # self.tokenizer.get_added_vocab() --> {'<pad>': 0, '<unk>': 1, '<bos>': 2, '<eos>': 3, '<an>': 4, '<en>': 5}
        assert result["target_ids"][0][0].item() == self.tokenizer.convert_tokens_to_ids("<en>")

        for i, tok_id in enumerate(result["target_ids"][0]):
            if i == 0:
                assert tok_id == self.tokenizer.convert_tokens_to_ids("<en>")
            elif i == len(result["target_ids"][0]) - 1:
                assert tok_id == self.tokenizer.eos_token_id
            else:
                assert tok_id == tgt_toks[i - 1]

        target_sequence = self.tokenizer.decode(result["target_ids"][0])
        assert target_sequence.endswith("<eos>")
        assert target_sequence.startswith("<en>")

    def test_collate_fn_no_lang_ids_bilingual(self):
        # Test that src/tgt_lang_ids have all [-1]
        for result in self.dm.val_dataloader():
            for item in result["source_lang_ids"]:
                assert item.item() == -1
            for item in result["target_lang_ids"]:
                assert item.item() == -1

    def test_collate_fn_lang_ids_multilingual(self):
        # Test that src/tgt_lang_ids have the proper lang ids
        for result in self.dm_with_lang_tags.val_dataloader():
            for item in result["source_lang_ids"]:
                assert item.item() == self.tokenizer.convert_tokens_to_ids("<an>") == 4
            for item in result["target_lang_ids"]:
                assert item.item() == self.tokenizer.convert_tokens_to_ids("<en>") == 5
        #TODO Test a case where there are multiple source / target languages.

    def test_collate_fn_no_lang_tags_in_sequence_bilingual(self):
        for result in self.dm.val_dataloader():
            for item in result["source_ids"]:
                assert item[0].item() not in self.tokenizer.get_added_vocab().values()
                assert item[0].item() not in self.tokenizer.all_special_ids
            for item in result["target_ids"]:
                assert item[0].item() not in self.tokenizer.get_added_vocab().values()
                assert item[0].item() not in self.tokenizer.all_special_ids

    def test_collate_fn_lang_tags_in_sequence_multilingual(self):
        for result in self.dm_with_lang_tags.val_dataloader():
            for item in result["source_ids"]:
                assert item[0].item() == self.tokenizer.convert_tokens_to_ids("<an>") == 4
            for item in result["target_ids"]:
                assert item[0].item() == self.tokenizer.convert_tokens_to_ids("<en>") == 5
        #TODO Test a case where there are multiple source / target languages

