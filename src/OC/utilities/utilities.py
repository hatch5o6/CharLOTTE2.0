import random

# split
def split(
    data: list, # list of tuples (freq, src, tgt, NLD).
    val_ratio: float,
    seed: int,
    min_val_size: int=250,
    max_val_size: int=1100,
    verbose=True
):
    if val_ratio > 1:
        raise ValueError(f"Val ratio out of bounds. Must be 0 <= val_ratio <= 1.")
    
    random.seed(seed)
    random.shuffle(data)
    val_amount = round(val_ratio * len(data))
    val_amount = _bind(val_amount, max_val_size, min_val_size)

    val = data[:val_amount]
    train = data[val_amount:]

    if verbose:
        print("VAL BEFORE SRC-SIDE UNIQUE:", len(val))
    val = _source_side_unique(val)

    if verbose:
        print("VAL:", len(val))
        print("TRAIN:", len(train))

    return val, train

def _bind(amount, max_size, min_size):
    amount = max(min_size, amount)
    amount = min(max_size, amount)
    return amount

def _source_side_unique(data):
    unique = {}
    for freq, src, tgt, nld in data:
        if src not in unique:
            unique[src] = tgt, freq, nld
    return [(freq, src, tgt, nld) for src, (tgt, freq, nld) in unique.items()]


# write
def write_oc_data(dataset, path):
    with open(path, "w") as outf:
        for row in dataset:
            outf.write(" ||| ".join([str(element) for element in row]) + "\n")