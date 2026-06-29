import os
from sloth_hatch import sloth

def parallel_format(directory, pair=("es", "an"), div="train"):
    assert div in ["train", "val"]
    pl, cl = pair
    train_pl = sloth.read_lines(os.path.join(directory, f"{div}.{pl}.txt"))
    train_cl = sloth.read_lines(os.path.join(directory, f"{div}.{cl}.txt"))
    assert len(train_pl) == len(train_cl)
    lines = []
    for pl_word, cl_word in zip(train_pl, train_cl):
        lines.append(f"0 ||| {pl_word} ||| {cl_word} ||| 0.0")
    sloth.write_lines(lines, os.path.join(directory, f"{div}.parallel.txt"))

def monolingual_format(directory, pair=("es", "an"), div="train"):
    assert div in ["train", "val"]
    pl, cl = pair
    train_pl = sloth.read_lines(os.path.join(directory, f"{div}.{pl}.txt"))
    train_cl = sloth.read_lines(os.path.join(directory, f"{div}.{cl}.txt"))
    assert len(train_pl) == len(train_cl)
    lines = []
    for pl_word, cl_word in zip(train_pl, train_cl):
        lines.append(f"0 ||| 0 ||| {pl_word} ||| {cl_word} ||| 0.0")
    sloth.write_lines(lines, os.path.join(directory, f"{div}.monolingual.txt"))

if __name__ == "__main__":
    d = "src/OC/train/tests/fixtures/train"
    parallel_format(d, div="train")
    parallel_format(d, div="val")
    monolingual_format(d, div="train")
    monolingual_format(d, div="val")
    