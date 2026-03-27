import argparse
from statistics import geometric_mean
from OC.utilities.utilities import write_oc_data

from sloth_hatch.sloth import log_parsed_args

def find_val_set(cognate_files, val_file, size=900):
    cognate_lists = [
        read_cognate_file(f)
        for f in cognate_files
    ]
    cognates_in_common = list(set.intersection(*[
        set(cognate_list.keys())
        for cognate_list in cognate_lists
    ]))
    for i, (src_word, tgt_word) in enumerate(cognates_in_common):
        freq_scores = [
            cognate_list[(src_word, tgt_word)][0]
            for cognate_list in cognate_lists
        ]
        # make it the frequency scores from each list, then the src_word, then the tgt_word, then the nld
        # freqs ..., src_word, tgt_word, nld
        cognates_in_common[i] = freq_scores + [src_word, tgt_word] + [cognate_lists[0][(src_word, tgt_word)][-1]]
    cognates_in_common.sort(reverse=True)

    val = cognates_in_common[:size]
    cognate_pairs_in_val = set(
        # (src_word, tgt_word)
        (cognates_and_scores[-3], cognates_and_scores[-2]) 
        for cognates_and_scores in val
    )
    cognates_in_common_without_scores = set(
        # (src_word, tgt_word)
        (cognates_and_scores[-3], cognates_and_scores[-2]) 
        for cognates_and_scores in cognates_in_common
    )
    
    train = []
    for cognate_list in cognate_lists:
        cognate_list_train = []
        for (src_word, tgt_word), scores in cognate_list.items():
            if (src_word, tgt_word) in cognate_pairs_in_val: continue
            if len(scores) == 2:
                freq, nld = scores
                cognate_list_train.append((freq, src_word, tgt_word, nld))
            else:
                assert len(scores) == 4
                geo_mean, src_freq, tgt_freq, nld = scores
                cognate_list_train.append((geo_mean, src_freq, tgt_freq, src_word, tgt_word, nld))
        train.append(cognate_list_train)
    
    cognates_in_common.sort(key=lambda x: x[-1])
    val.sort(key=lambda x: x[-1])

    print("Cognates in common:", len(cognates_in_common))
    write_oc_data(cognates_in_common, val_file + ".all_in_common")
    print("Val:", len(val))
    write_oc_data(val, val_file)

    train_paths = []
    for i, cognate_list_train in enumerate(train):
        cognate_list_train.sort(key=lambda x: x[-1])
        print(f"Train {i}:", len(cognate_list_train))

        tuple_len = len(cognate_list_train[0])
        assert tuple_len in [4, 6]
        for item in cognate_list_train:
            assert len(item) == tuple_len

        path = cognate_files[i] + ".train"
        train_paths.append(path)
        write_oc_data(cognate_list_train, path)
    
    monolingual_cognates_in_common = []
    for (src_word, tgt_word), \
        (geo_mean, src_freq, tgt_freq, nld) \
        in cognate_lists[-1].items():
        if (src_word, tgt_word) in cognates_in_common_without_scores:
            monolingual_cognates_in_common.append(
                (geo_mean, src_freq, tgt_freq, src_word, tgt_word, nld)
            )
    write_oc_data(monolingual_cognates_in_common, val_file + ".monolingual_cognates_in_common")

    return val_file, train_paths

def read_cognate_file(f):
    with open(f) as inf:
        cognates = [l.strip().split(" ||| ") for l in inf.readlines()]
    if len(cognates[0]) == 4:
        return parallel_cognates(cognates)
    else:
        assert len(cognates[0]) == 5
        return monolingual_cognates(cognates)

def parallel_cognates(cognates):
    return {
        (src_word, tgt_word): (int(freq), float(nld))
        for freq, src_word, tgt_word, nld in cognates
    }

def monolingual_cognates(cognates):
    return {
        (src_word, tgt_word): (geometric_mean([int(src_freq), int(tgt_freq)]), int(src_freq), int(tgt_freq), float(nld))
        for src_freq, tgt_freq, src_word, tgt_word, nld in cognates
    }

@log_parsed_args
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--cognate_files", help="comma-delimited list of paths to read")
    parser.add_argument("-v", "--val_file", help="path to write validation file")
    parser.add_argument("-s", "--size", type=int, default=900)
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    cognate_files = [f.strip() for f in args.cognate_files.split(",")]
    find_val_set(
        cognate_files=cognate_files,
        val_file=args.val_file,
        size=args.size
    )
