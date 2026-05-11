from torch.utils.data import Dataset
import os
from lightning.pytorch.utilities import rank_zero_info
from sloth_hatch.sloth import read_lines

from utilities.utilities import set_vars_in_path

class CharLOTTEParallelDataset(Dataset):
    def __init__(
        self,
        datasets:list=[], # list of [data folder, pl, cl, tl] tuples
        sc_model_ids:dict=None,
        reverse:bool=False,
        mode:str="parent",
        div:str="train",
        inference_file=None,
        inference_src=None,
        inference_tgt=None
    ):
        for item in datasets:
            if not (isinstance(item, list) or isinstance(item, tuple)):
                raise ValueError("datasets must be a list of tuples/lists!")
            if len(item) != 4:
                raise ValueError("Each item in datasets must be a tuple/list of length 4: data folder, pl, cl, tl")
        if mode not in ["parent", "child", "oc"]:
            raise ValueError(f"mode must be 'parent', 'child', or 'oc'")
        if sc_model_ids and mode != "parent":
            raise ValueError("Can only pass sc_model_ids when mode='parent'!")
        if div not in ["train", "val", "test"]:
            raise ValueError(f"div must be 'train', 'val', or 'test'")
        if inference_file:
            if not isinstance(inference_file, str):
                raise ValueError("inference_file must be a file path!")
            if len(datasets) > 0:
                raise ValueError("datasets must be an empty list if passing inference_file!")

        if inference_file:
            if not isinstance(inference_src, str):
                raise ValueError("Must pass a string as inference_src!")
            if not isinstance(inference_tgt, str):
                raise ValueError("Must pass a string as inference_tgt!")
            self.inference_file = inference_file
            self.inference_src = inference_src
            self.inference_tgt = inference_tgt

            self.data = self._read_inference_file()
        else:
            self.datasets = datasets
            self.sc_model_ids = sc_model_ids
            self.reverse = reverse
            self.mode = mode
            self.div = div

            self.data = self._read_data()

    def _read_inference_file(self):
        data = []
        lines = read_lines(self.inference_file)
        for line in lines:
            data.append({
                "src_lang": self.inference_src,
                "tgt_lang": self.inference_tgt,
                "src_path": self.inference_file,
                "tgt_path": "N/A",
                "src": line,
                "tgt": "<to be generated>"
            })
        return data

    def _read_data(self):
        data = []
        for data_folder, pl, cl, tl in self.datasets:
            data_folder = set_vars_in_path(data_folder)

            if self.mode == "parent":
                src_lang = pl
                tgt_lang = tl
            elif self.mode == "child":
                src_lang = cl
                tgt_lang = tl
            elif self.mode == "oc":
                src_lang = pl
                tgt_lang = cl
            else:
                assert False

            directory = os.path.join(data_folder, f"{src_lang}-{tgt_lang}")
            src_path = os.path.join(directory, f"{self.div}.{src_lang}.txt")
            tgt_path = os.path.join(directory, f"{self.div}.{tgt_lang}.txt")

            if self.sc_model_ids:
                assert self.mode == "parent"
                if (pl, cl, tl) not in self.sc_model_ids:
                    raise ValueError(f"({pl}, {cl}, {tl}) missing from sc_model_ids!")
                sc_model_id = self.sc_model_ids[(pl, cl, tl)]
                src_path = f"{src_path}.{sc_model_id}"
            
            if self.mode == "parent":
                assert src_lang == pl
                src_lang = cl

            if self.reverse:
                src_lang, tgt_lang = tgt_lang, src_lang
                src_path, tgt_path = tgt_path, src_path
            
            src_lines = read_lines(src_path)
            tgt_lines = read_lines(tgt_path)
            if len(src_lines) != len(tgt_lines):
                raise ValueError(f"Length of source lines != length of target lines!:\n\t-src:`{src_path}`\n\t-tgt:`{tgt_path}`")

            for src_line, tgt_line in zip(src_lines, tgt_lines):
                data.append({
                    "src_lang": src_lang,
                    "tgt_lang": tgt_lang,
                    "src_path": src_path,
                    "tgt_path": tgt_path,
                    "src": src_line,
                    "tgt": tgt_line
                })
        return data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]

def print_dataset(dataset, limit=None, start=None):
    print("len:", len(dataset))
    for i in range(len(dataset)):
        if start and i < start:
            continue
        print(f"{i}) {dataset[i]}")
        if limit and i == limit:
            break

if __name__ == "__main__":
    from sloth_hatch.sloth import read_yaml
    config = read_yaml("src/configs/test.yaml")
    print("DATA:", config["data"])
    div = "test"
    # parent_dataset = CharLOTTEParallelDataset(
    #     datasets=config["data"],
    #     sc_model_ids=None,
    #     mode="parent",
    #     div=div,
    #     reverse=True
    # )
    # print("PARENT")
    # print_dataset(parent_dataset, limit=3)
    # print("\n\n")
    # child_dataset = CharLOTTEParallelDataset(
    #     datasets=config["data"],
    #     sc_model_ids=None,
    #     mode="child",
    #     div=div
    # )
    # print("CHILD")
    # print_dataset(child_dataset, limit=3)
    # print("\n\n")
    # oc_dataset = CharLOTTEParallelDataset(
    #     datasets=config["data"],
    #     sc_model_ids=None,
    #     mode="oc",
    #     div=div
    # )
    # print("OC")
    # print_dataset(oc_dataset, limit=3)
    # print("\n\n")

