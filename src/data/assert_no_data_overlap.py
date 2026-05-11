import argparse
import os
import shutil
import csv
import json

def set_env():
    assert os.path.exists(".env"), f"NO .env FILE FOUND"
    with open(".env") as env_file:
        for line in env_file:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, val = line.partition("=")
                os.environ.setdefault(key.strip(), val.strip())

def assert_no_overlap(
    train,
    dev,
    test,
    VERBOSE
):
    """
    train/dev/test: list of (src_line, tgt_line) tuples
    """

    train_src, train_tgt = split_src_tgt(train)
    dev_src, dev_tgt = split_src_tgt(dev)
    test_src, test_tgt = split_src_tgt(test)

    results = {}

    if VERBOSE:
        print("\nTRAIN PAIR")
        print_data(train)
        print("\nDEV PAIR")
        print_data(dev)
        print("\nTEST PAIR")
        print_data(test)
    results.update(assert_sets(
        train_set=set(train),
        dev_set=set(dev),
        test_set=set(test),
        VERBOSE=True,
        TAG="PAIR"
    ))

    if VERBOSE:
        print("\nTRAIN SRC")
        print_data(train_src)
        print("\nDEV SRC")
        print_data(dev_src)
        print("\nTEST SRC")
        print_data(test_src)
    results.update(assert_sets(
        train_set=set(train_src),
        dev_set=set(dev_src),
        test_set=set(test_src),
        VERBOSE=True,
        TAG="SRC"
    ))

    if VERBOSE:
        print("\nTRAIN TGT")
        print_data(train_tgt)
        print("\nDEV TGT")
        print_data(dev_tgt)
        print("\nTEST TGT")
        print_data(test_tgt)
    results.update(assert_sets(
        train_set=set(train_tgt),
        dev_set=set(dev_tgt),
        test_set=set(test_tgt),
        VERBOSE=True,
        TAG="TGT"
    ))

    print("TEST RESULTS:")
    print(json.dumps(results, ensure_ascii=False, indent=2))
    if all(list(results.values())):
        print("passed!\n\n")
        return True, results
    else:
        print("failed!\n\n")
        return False, results

def read_csv_by_file(csv_f):
    set_env()
    with open(csv_f, newline='') as inf:
        rows = [r for r in csv.reader(inf)]
    header = rows[0]
    assert header == ["src_lang", "tgt_lang", "src_path", "tgt_path"]
    data = [tuple(r) for r in rows[1:]]

    data_dicts = []
    all_src = []
    all_tgt = []
    for src_lang, tgt_lang, src_path, tgt_path in data:
        src_path = os.path.expandvars(src_path)
        tgt_path = os.path.expandvars(tgt_path)

        src_lines = read_f(src_path)
        tgt_lines = read_f(tgt_path)
        data_dicts.append({
            "src_lang": src_lang,
            "tgt_lang": tgt_lang,
            "src_path": src_path,
            "tgt_path": tgt_path,
            "pairs": list(zip(src_lines, tgt_lines))
        })

        all_src += src_lines
        all_tgt += tgt_lines
    return data_dicts, set(all_src), set(all_tgt)


def make_fixes(train_csv, dev_csv, test_csv):
    train_dicts, train_src_set, train_tgt_set = read_csv_by_file(train_csv)
    dev_dicts, dev_src_set, dev_tgt_set = read_csv_by_file(dev_csv)
    test_dicts, test_src_set, test_tgt_set = read_csv_by_file(test_csv)

    for t, train_row in enumerate(train_dicts):
        new_train_row_pairs = []
        for idx, (train_src_line, train_tgt_line) in enumerate(train_row["pairs"]):
            REMOVE = any([
                train_src_line in dev_src_set,
                train_src_line in test_src_set,

                train_tgt_line in dev_tgt_set,
                train_tgt_line in test_tgt_set
            ])
            if not REMOVE:
                new_train_row_pairs.append((train_src_line, train_tgt_line))
        train_row["new_pairs"] = new_train_row_pairs
        train_dicts[t] = train_row

    for d, dev_row in enumerate(dev_dicts):
        new_dev_row_pairs = []
        for idx, (dev_src_line, dev_tgt_line) in enumerate(dev_row["pairs"]):
            REMOVE = any([
                dev_src_line in test_src_set,

                dev_tgt_line in test_tgt_set
            ])
            if not REMOVE:
                new_dev_row_pairs.append((dev_src_line, dev_tgt_line))
        dev_row["new_pairs"] = new_dev_row_pairs
        dev_dicts[d] = dev_row

    return train_dicts, dev_dicts
    

def assert_sets(
    train_set: set, 
    dev_set: set,
    test_set: set,
    VERBOSE: bool = False,
    TAG: str = "PAIR"
):
    intersection = train_set.intersection(dev_set, test_set)
    train_dev_intersect = train_set.intersection(dev_set)
    train_test_intersect = train_set.intersection(test_set)
    dev_test_intersect = dev_set.intersection(test_set)
    if VERBOSE:
        print(f"{TAG} intersection")
        print("\tlen:", len(intersection), sorted(list(intersection)))
        print(f"{TAG} train-dev intersection")
        print("\tlen:", len(train_dev_intersect), sorted(list(train_dev_intersect)))
        print(f"{TAG} train-test intersection")
        print("\tlen:", len(train_test_intersect), sorted(list(train_test_intersect)))
        print(f"{TAG} dev-test intersection")
        print("\tlen:", len(dev_test_intersect), sorted(list(dev_test_intersect)))
    
    return {
        f"{TAG}_intersection": len(intersection) == 0,
        f"{TAG}_train_dev_intersect": len(train_dev_intersect) == 0,
        f"{TAG}_train_test_intersect": len(train_test_intersect) == 0,
        f"{TAG}_dev_test_intersect": len(dev_test_intersect) == 0
    }
    

def print_data(data, LIMIT=20):
    for i, item in enumerate(data):
        print(f"\t({i + 1}) -`{item}`")
        if i == LIMIT:
            break

def split_src_tgt(pairs):
    src_segs = [src for src, tgt in pairs]
    tgt_segs = [tgt for src, tgt in pairs]
    return src_segs, tgt_segs
    
def read_csv(f, sc_model_id=None):
    with open(f, newline='') as inf:
        rows = [r for r in csv.reader(inf)]
    header = rows[0]
    assert header == ["src_lang", "tgt_lang", "src_path", "tgt_path"]
    rows = rows[1:]
    rows = [tuple(r) for r in rows]
    src_lines = []
    tgt_lines = []
    for src_lang, tgt_lang, src_path, tgt_path in rows:
        if "SC_{SC_MODEL_ID}" in src_path:
            assert sc_model_id != None
            assert src_path.endswith(f"_{src_lang}2{tgt_lang}.txt")
            src_path = src_path.replace("{SC_MODEL_ID}", sc_model_id)

        src_lines += read_f(src_path)
        tgt_lines += read_f(tgt_path)
    assert len(src_lines) == len(tgt_lines)
    pairs = list(zip(src_lines, tgt_lines))
    return pairs
    
def read_f(f):
    with open(f) as inf:
        lines = [l.strip() for l in inf.readlines()]
    return lines

def test_data_dirs(
    data_dirs,
    HAVE_TAG,
    INCLUDES_DEV_TEST,
    PLAIN_dir,
    MAKE_FIXES,
    MAKE_FIXES_TAG
):
    print("INCLUDES_DEV_TEST =", INCLUDES_DEV_TEST)
    lang_pairs = {}
    for d in os.listdir(data_dirs):
        lp = d.split("_dev_test")[0]
        srcl, tgtl = lp.split("-")
        assert srcl.strip() != ""
        assert tgtl.strip() != ""
        if INCLUDES_DEV_TEST:
            assert d in [lp, f"{lp}_dev_test"]
        else:
            assert d == lp
        if lp not in lang_pairs:
            lang_pairs[lp] = []
        lang_pairs[lp].append(d)
    
    if INCLUDES_DEV_TEST:
        assert len(os.listdir(data_dirs)) == len(lang_pairs) * 2
    else:
        assert len(os.listdir(data_dirs)) == len(lang_pairs)

    lp_scores = {}
    for lp, dirs in lang_pairs.items():
        srcl, tgtl = lp.split("-")

        print("SRC:", srcl)
        print("TGT:", tgtl)

        print(f"\n\n####################################### LP: {lp} #######################################")
        if INCLUDES_DEV_TEST:
            assert set(dirs) == {lp, f"{lp}_dev_test"}
        else:
            assert set(dirs) == {lp}

        print(f"{lp} FILES:")
        if HAVE_TAG is not None:
            file_tag = f".{HAVE_TAG}"
        else:
            file_tag = ""

        train_csv = os.path.join(data_dirs, lp, f"train{file_tag}.csv")
        if INCLUDES_DEV_TEST:
            val_csv = os.path.join(data_dirs, f"{lp}_dev_test", f"val{file_tag}.csv")
            test_csv =  os.path.join(data_dirs, f"{lp}_dev_test", f"test.csv") # Test data should NEVER be altered, so we don't need a file_tag for a different version, as they will not exist.
        else:
            val_csv = os.path.join(PLAIN_dir, f"{lp}_dev_test", f"val{file_tag}.csv")
            test_csv =  os.path.join(PLAIN_dir, f"{lp}_dev_test", f"test.csv")

        print("asserting train langs")
        assert_langs(srcl, tgtl, train_csv)
        print("asserting val langs")
        assert_langs(srcl, tgtl, val_csv)
        print("asserting test langs")
        assert_langs(srcl, tgtl, test_csv)

        VERBOSE = lp in ["an-en"]
        if VERBOSE:
            print(lp, "VERBOSE = True")

        print(f"Asserting no overlap between\n\t-train:{train_csv}\n\t-dev:{val_csv}\n\t-test:{test_csv}\n")
        train = read_csv(train_csv)
        val = read_csv(val_csv)
        test = read_csv(test_csv)
        print("ASSERTING NO OVERLAP")
        passed, assert_results = assert_no_overlap(
            train=train,
            dev=val,
            test=test,
            VERBOSE=VERBOSE
        )
        assert lp not in lp_scores
        lp_scores[lp] = passed
        
        train_set = set(train)
        val_set = set(val)
        if MAKE_FIXES:
            print("Making Fixes")
            train_dicts, dev_dicts = make_fixes(
                train_csv=train_csv,
                dev_csv=val_csv,
                test_csv=test_csv
            )
            print("TRAIN REDUCTIONS:")
            all_new_train = []
            for train_row in train_dicts:
                print("\t", train_row["src_lang"], train_row["tgt_lang"], train_row["src_path"], train_row["tgt_path"])
                print("\t\t", len(train_row["pairs"]), ">", len(train_row["new_pairs"]))
                print("\t\tasserting each new pair is in the original (should be, because we only removed from the original to create the new)")
                all_new_train += train_row["new_pairs"]
                for pair in train_row["new_pairs"]:
                    assert pair in train_set
                print("\t\tpassed")
            print("DEV REDUCTIONS")
            all_new_val = []
            for dev_row in dev_dicts:
                print("\t", dev_row["src_lang"], dev_row["tgt_lang"], dev_row["src_path"], dev_row["tgt_path"])
                print("\t\t", len(dev_row["pairs"]), ">", len(dev_row["new_pairs"]))
                print("\t\tasserting each new pair is in the original (should be, because we only removed from the original to create the new)")
                all_new_val += dev_row["new_pairs"]
                for pair in dev_row["new_pairs"]:
                    assert pair in val_set
                print("\t\tpassed")
            
            print("\nASSERTING NO OVERLAP IN NEW FILES")
            new_passed, new_assert_results = assert_no_overlap(
                train=all_new_train,
                dev=all_new_val,
                test=test,
                VERBOSE=True
            )
            assert f"new_{lp}" not in lp_scores
            lp_scores[f"new_{lp}"] = new_passed
            
            train_dicts = write_new_files(train_dicts, tag=MAKE_FIXES_TAG)
            dev_dicts = write_new_files(dev_dicts, tag=MAKE_FIXES_TAG)

            print("Writing new train csv file")
            write_new_csv(
                dicts=train_dicts,
                old_csv_f=train_csv,
                tag=MAKE_FIXES_TAG
            )
            write_new_csv(
                dicts=dev_dicts,
                old_csv_f=val_csv,
                tag=MAKE_FIXES_TAG
            )
            
    print("SUMMARY:")
    print(json.dumps(lp_scores, ensure_ascii=False, indent=2))


def assert_langs(srcl, tgtl, csvf):
    with open(csvf, newline='') as inf:
        rows = [r for r in csv.reader(inf)]
    header = rows[0]
    assert header == ["src_lang", "tgt_lang", "src_path", "tgt_path"]
    rows = rows[1:]
    rows = [tuple(r) for r in rows]
    for src_lang, tgt_lang, src_path, tgt_path in rows:
        assert srcl == src_lang
        assert tgtl == tgt_lang

        
def write_new_files(dicts, tag="no_overlap"):
    written_to_paths = set()
    for r, row in enumerate(dicts):
        new_src_path = get_new_path(row["src_path"], tag=tag)
        new_tgt_path = get_new_path(row["tgt_path"], tag=tag)
        assert new_src_path != new_tgt_path
        assert new_src_path not in written_to_paths
        assert new_tgt_path not in written_to_paths
        written_to_paths.add(new_src_path)
        written_to_paths.add(new_tgt_path)
        assert "new_src_path" not in row
        row["new_src_path"] = new_src_path
        assert "new_tgt_path" not in row
        row["new_tgt_path"] = new_tgt_path
        dicts[r] = row
        write_pairs(row["new_pairs"], src_f=new_src_path, tgt_f=new_tgt_path)
    return dicts

def get_new_path(og_path, tag):
    ext = og_path.split(".")[-1]
    assert ext.strip() != ""
    new_path = og_path[:-len(ext)] + f"{tag}.{ext}"
    assert new_path != og_path
    return new_path

def write_pairs(pairs, src_f, tgt_f):
    with open(src_f, "w") as srcf, open(tgt_f, "w") as tgtf:
        for src_line, tgt_line in pairs:
            srcf.write(src_line.strip() + "\n")
            tgtf.write(tgt_line.strip() + "\n")


def write_new_csv(
    dicts,
    old_csv_f,
    tag
):
    new_csv_f = get_new_path(old_csv_f, tag=tag)
    header = ["src_lang", "tgt_lang", "src_path", "tgt_path"]
    with open(new_csv_f, "w", newline='') as outf:
        writer = csv.writer(outf)
        writer.writerow(header)
        for item in dicts:
            row = [item["src_lang"], item["tgt_lang"], item["new_src_path"], item["new_tgt_path"]]
            writer.writerow(row)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="NMT/data/PLAIN")
    parser.add_argument("--HAVE_TAG", default=None, help="if passed, then will read train, val, and test files with tag, as in train.{tag}.csv (instead of train.csv)")
    parser.add_argument("--INCLUDES_DEV_TEST", default=False, action="store_true", help="if passed, then tells it that val and test are in --dir")
    parser.add_argument("--PLAIN_dir", default=None, help="for when NOT passing INCLUDES_DEV_TEST (i.e. when doing augmented data). This tells it where to get val / test")
    parser.add_argument("--MAKE_FIXES", default=False, action="store_true", help="If passed, will create new train and val files so that train, val, and test have no overlapping src lines or tgt lines, and then create a new data .csv.")
    parser.add_argument("--MAKE_FIXES_TAG", default="no_overlap", help="the tag name to append to newly created files when passing --MAKE_FIXES")
    args = parser.parse_args()
    print("Arguments:")
    for k, v in vars(args).items():
        print(f"--{k}=`{v}`")
    print("-----------------\n\n\n")
    return args

if __name__ == "__main__":
    print("------------------------------------------------------------------------------")
    print("#############################")
    print("# assert_no_data_overlap.py #")
    print("#############################")
    args = get_args()
    if args.INCLUDES_DEV_TEST == False:
        assert args.PLAIN_dir is not None
    if args.PLAIN_dir is None:
        assert args.INCLUDES_DEV_TEST == True

    test_data_dirs(
        data_dirs=args.dir, 
        HAVE_TAG=args.HAVE_TAG,
        INCLUDES_DEV_TEST=args.INCLUDES_DEV_TEST, 
        PLAIN_dir=args.PLAIN_dir, 
        MAKE_FIXES=args.MAKE_FIXES,
        MAKE_FIXES_TAG=args.MAKE_FIXES_TAG
    )