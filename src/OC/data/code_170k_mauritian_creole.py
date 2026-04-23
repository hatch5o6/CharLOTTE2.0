# From https://huggingface.co/datasets/michsethowusu/Code-170k-mauritian-creole
"""
@dataset{code170k_mauritian_creole,
  title={Code-170k-mauritian-creole: Programming Conversations in Mauritian Creole},
  year={2025},
  publisher={Hugging Face},
  url={https://huggingface.co/datasets/michsethowusu/Code-170k-mauritian-creole}
}
"""
WRITE_TO = "/home/hatch5o6/nobackup/archive/data/MONOLINGUAL/MauritianCreole/Sources/Code170k/mfe.txt"

from datasets import load_dataset
from tqdm import tqdm

# Load the dataset
dataset = load_dataset("michsethowusu/Code-170k-mauritian-creole")

# Access training data
train_data = dataset['train']

print(train_data)

# Example: Print first conversation
with open(WRITE_TO, "w") as outf:
    for item in tqdm(train_data):
        # print("ITEM:", item)
        for turn in item['conversations']:
            # print("TURN:", turn)
            outf.write(turn["value"].strip() + "\n")

# for turn in train_data[0]['conversations']:
#     print(f"{turn['from']}: {turn['value']}")

