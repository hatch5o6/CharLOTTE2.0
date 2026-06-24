import argparse
from assert_no_data_overlap import assert_no_overlap
import copy
import json
import os
import shutil
from utilities.utilities import set_env
import subprocess
import random
import argparse

set_env()
DATA_HOME = os.environ["DATA_HOME"]

out_dir = f"{DATA_HOME}/CharLOTTE_data"
raw_data = f"{DATA_HOME}/raw"

nllb=f"{raw_data}/NLLB"
oldi=f"{raw_data}/OLDI"
ccmat=f"{raw_data}/CCMatrix"
ccalign=f"{raw_data}/CCAligned"
wikimed=f"{raw_data}/wikimedia"
wikimat=f"{raw_data}/WikiMatrix"
wmt=f"{raw_data}/WMT20"
kreyolmt=f"{raw_data}/KreyolMT"
kreolmorisienmt=f"{raw_data}/KreolMorisienMT"
mt560=f"{raw_data}/MT560"
ldc=f"{raw_data}/LDC"
twb=f"{raw_data}/TWB"
chavmt=f"{raw_data}/ChavacanoMT"
dgt=f"{raw_data}/DGT"
hplt=f"{raw_data}/HPLT"
doda=f"{raw_data}/DODa"
flores=f"{raw_data}/flores+"
combined=f"{raw_data}/combined"


def main(
    src_train:tuple,
    tgt_train:tuple,

    src_val:str,
    tgt_val:str,

    src_test:str,
    tgt_test:str,

    src_lang:str,
    tgt_lang:str,
):
    assert src_train is not None
    assert tgt_train is not None
    assert src_lang is not None
    assert tgt_lang is not None

    if src_val is None:
        assert tgt_val is None
    if tgt_val is None:
        assert src_val is None
    
    if src_test is None:
        assert tgt_test is None
    if tgt_test is None:
        assert src_test is None

    print(f"########### {src_lang} / {tgt_lang} ###########")

    # OUT DIR
    pair_dir = f"{out_dir}/{src_lang}-{tgt_lang}"
    if os.path.exists(pair_dir):
        print("DELETING", pair_dir)
        shutil.rmtree(pair_dir)
    subprocess.call(["mkdir", "-p", pair_dir])

    # DEDUPE TRAINING DATA
    src_train = [src_train] if isinstance(src_train, str) else src_train
    tgt_train = [tgt_train] if isinstance(tgt_train, str) else tgt_train
    train = []
    for src_train_f, tgt_train_f in zip(src_train, tgt_train):
        print(src_train_f, tgt_train_f)
        train += get_pairs(src_train_f, tgt_train_f)
    print("\n\nDEDUPING THE TRAINING DATA")
    train = dedupe_data(train)

    val = get_pairs(src_val, tgt_val)
    test = get_pairs(src_test, tgt_test)

    print("TRAIN:", len(train))
    print("VAL:", len(val))
    print("TEST:", len(test))

    # ENSURE NO OVERLAP BETWEEN TRAIN / DEV / TEST
    print("\n\nREMOVING OVERLAP WITH VAL/TEST")
    new_train, new_val = remove_overlap(train, val, test)

    assert new_val == val # this shouldn't change for our data

    passed, results = assert_no_overlap(
        train=new_train,
        dev=new_val,
        test=test,
        VERBOSE=False
    )
    print(json.dumps(results, indent=2))

    assert passed == True
    print("LEN NEW TRAIN:", len(new_train))
    print("LEN NEW VAL:", len(new_val))
    print("LEN TEST:", len(test))

    # WRITE TO FILE
    # truncate at 10mil
    if len(new_train) > 10000000:
        new_train = new_train[:10000000]

    write_set(pair_dir, new_train, src_lang, tgt_lang, div="train")
    write_set(pair_dir, new_val, src_lang, tgt_lang, div="val")
    write_set(pair_dir, test, src_lang, tgt_lang, div="test")


def dedupe_data(data):
    print("BEFORE:", len(data))
    used = set()
    deduped = []
    for item in data:
        if item not in used:
            deduped.append(item)
            used.add(item)
    print("AFTER:", len(deduped))
    return deduped

def get_pairs(src_file, tgt_file):
    if src_file == None or tgt_file == None:
        return []

    print(f"READING PARALLEL DATA:\n\t-`{src_file}`\n\t-`{tgt_file}`")
    src_lines = read_file(src_file)
    tgt_lines = read_file(tgt_file)
    assert len(src_lines) == len(tgt_lines)
    return list(zip(src_lines, tgt_lines))

def read_file(f):
    with open(f) as inf:
        lines = [l.strip() for l in inf.readlines()]
    return lines

def write_set(pair_dir, pairs, src_lang, tgt_lang, div="train"):
    print(f"Writing {src_lang}-{tgt_lang} {div} to {pair_dir}")
    assert div in ["train", "test", "val"]
    src_path = os.path.join(pair_dir, f"{div}.{src_lang}.txt")
    tgt_path = os.path.join(pair_dir, f"{div}.{tgt_lang}.txt")
    write_pairs(src_path, tgt_path, pairs)

def write_pairs(src_f, tgt_f, pairs):
    with open(src_f, "w") as sf, open(tgt_f, "w") as tf:
        for src_line, tgt_line in pairs:
            sf.write(src_line.strip() + "\n")
            tf.write(tgt_line.strip() + "\n")

def remove_overlap(train_set, fine_tune_set, test_set):
    train_pairs = copy.deepcopy(train_set)
    fine_tune_pairs = copy.deepcopy(fine_tune_set)
    test_pairs = copy.deepcopy(test_set)

    fine_tune_src_set, fine_tune_tgt_set =      get_src_tgt_sets(fine_tune_pairs)
    test_src_set, test_tgt_set =                get_src_tgt_sets(test_pairs)

    new_train_pairs = []
    for train_src_seg, train_tgt_seg in train_pairs:
        REMOVE = False
        if any([
            train_src_seg in fine_tune_src_set,
            train_src_seg in test_src_set,

            train_tgt_seg in fine_tune_tgt_set,
            train_tgt_seg in test_tgt_set
        ]):
            REMOVE = True
        if not REMOVE:
            new_train_pairs.append((train_src_seg, train_tgt_seg))
    
    new_fine_tune_pairs = []
    for fine_tune_src_seg, fine_tune_tgt_seg in fine_tune_pairs:
        REMOVE = False
        if any([
            fine_tune_src_seg in test_src_set,

            fine_tune_tgt_seg in test_tgt_set
        ]):
            REMOVE = True
        if not REMOVE:
            new_fine_tune_pairs.append((fine_tune_src_seg, fine_tune_tgt_seg))
            
    return new_train_pairs, new_fine_tune_pairs

def get_src_tgt_sets(pairs):
    src_set = set([src for src, tgt in pairs])
    tgt_set = set([tgt for src, tgt in pairs])
    return src_set, tgt_set


def get_subset(pair_path, src_lang, tgt_lang, div, n, seed=12):
    src_f, tgt_f = f"{pair_path}/cleaned/src.txt", f"{pair_path}/cleaned/tgt.txt"
    
    src_lines, tgt_lines = read_file(src_f), read_file(tgt_f)

    assert len(src_lines) == len(tgt_lines), "files have different line counts"
    
    deduped_lines = dedupe_data(list(zip(src_lines, tgt_lines)))
    src_lines, tgt_lines = zip(*deduped_lines)
    src_lines, tgt_lines = list(src_lines), list(tgt_lines)

    # collect src/tgt sets from any existing splits
    existing_src, existing_tgt = set(), set()
    for existing_div in ["val", "test"]:
        es_f = f"{pair_path}/{existing_div}/src.txt"
        et_f = f"{pair_path}/{existing_div}/tgt.txt"
        if os.path.exists(es_f) and os.path.exists(et_f):
            for s in read_file(es_f): existing_src.add(s)
            for t in read_file(et_f): existing_tgt.add(t)

    # filter candidates to those with no src/tgt overlap with existing splits
    candidates = [
        (i, s, t) for i, (s, t) in enumerate(zip(src_lines, tgt_lines))
        if s not in existing_src and t not in existing_tgt
    ]

    assert n <= len(candidates), f"not enough non-overlapping candidates: {len(candidates)} < {n}"

    rng = random.Random(seed)
    sampled = rng.sample(candidates, n)
    sampled_indices = set(i for i, _, _ in sampled)

    subprocess.call(["mkdir", "-p", f"{pair_path}/{div}/"])
    with open(f"{pair_path}/{div}/src.txt", "w") as src_out, \
         open(f"{pair_path}/{div}/tgt.txt", "w") as tgt_out:
        for _, s, t in sorted(sampled, key=lambda x: x[0]):
            src_out.write(s + "\n")
            tgt_out.write(t + "\n")

    remainder = [i for i in range(len(src_lines)) if i not in sampled_indices]
    with open(f"{pair_path}/cleaned/src.txt", "w") as src_orig, \
         open(f"{pair_path}/cleaned/tgt.txt", "w") as tgt_orig:
        for i in remainder:
            src_orig.write(src_lines[i] + "\n")
            tgt_orig.write(tgt_lines[i] + "\n")
    return

def get_args():
    parser = argparse.ArgumentParser(description="Clean Language Scenarios")

    SCENARIOS = ['es/pt-en', 'es/an-en', 'fr/mfe-en', 'uz/kaa-en', 'cs/hsb-de', 
                'am/ti-en', 'tl/bik-en', 'bn/rhg-en', 'mt/aeb-en', 'mt/ary-en', 
                'fr/crs-en', 'ca/oc-en', 'es/cbk-en']

    parser.add_argument('--language_scenario', '-l', type=str, nargs='+', 
                        choices=SCENARIOS, default=SCENARIOS)

    return parser.parse_args()

def check_lang_pair(src, tgt):
    exists = True
    for split in ["test", "train", "val"]:
        if not os.path.exists(f"{out_dir}/{src}-{tgt}/{split}.{src}.txt"):
            exists = False
        if not os.path.exists(f"{out_dir}/{src}-{tgt}/{split}.{tgt}.txt"):
            exists = False
    return exists

def check_subset(dataset, src, tgt, split):
    return os.path.exists(f"{dataset}/{src}_{tgt}/{split}/src.txt") and os.path.exists(f"{dataset}/{src}_{tgt}/{split}/tgt.txt")
        

def combine_datasets(dataset1, dataset2, dataset3, src, tgt):
    lang_pair = src + '_' + tgt
    src_1_f = f"{dataset1}/{lang_pair}/cleaned/src.txt"
    tgt_1_f = f"{dataset1}/{lang_pair}/cleaned/tgt.txt"

    src_2_f = f"{dataset2}/{lang_pair}/cleaned/src.txt"
    tgt_2_f = f"{dataset2}/{lang_pair}/cleaned/tgt.txt"

    src_3_f = f"{dataset3}/{lang_pair}/cleaned/src.txt"
    tgt_3_f = f"{dataset3}/{lang_pair}/cleaned/tgt.txt"

    pairs = []
    
    d1 = get_pairs(src_1_f, tgt_1_f)
    d1 = dedupe_data(d1)

    d2 = get_pairs(src_2_f, tgt_2_f)
    d2 = dedupe_data(d2)

    d3 = get_pairs(src_3_f, tgt_3_f)
    d3 = dedupe_data(d3)

    print(f"Dataset 1: {len(d1)}")
    
    pairs += d1
    if len(pairs) < 10000000: 
        pairs += d2[:10000000 - len(pairs)]
        pairs = dedupe_data(pairs)
        print(f"Dataset 2: {len(pairs) - len(d1)}")
    if len(pairs) < 11000000: # 10.1 mil to be safe with deduping
        print(f"Dataset 3: {10000000 - len(pairs)}")
        pairs += d3[:11000000 - len(pairs)]

    assert len(pairs) == 11000000

    subprocess.call(['mkdir', '-p', f"{combined}/{lang_pair}"])
    with open(f"{combined}/{lang_pair}/src.txt", "w") as out_src, \
         open(f"{combined}/{lang_pair}/tgt.txt", "w") as out_tgt:
        for src_line, tgt_line in pairs:
            out_src.write(src_line + '\n')
            out_tgt.write(tgt_line + '\n')
        
    return

def combine_test_val_sets_aeb_en():
    # used to deal with ldc duplicates and extra test files

    aeb_en = f"{ldc}/aeb_en"
    src_val_f, src_tgt_f = f"{aeb_en}/dev/src.txt", f"{aeb_en}/dev/tgt.txt"
    val_data = get_pairs(src_val_f, src_tgt_f)
    val_data = dedupe_data(val_data)
    
    test1_src_f, test1_tgt_f = f"{aeb_en}/test1/src.txt", f"{aeb_en}/test1/tgt.txt"
    test1_data = get_pairs(test1_src_f, test1_tgt_f)
    test1_data = dedupe_data(test1_data)

    test2_data = []
    for split in ['test2', 'test3']:
        src_f, tgt_f = f"{aeb_en}/{split}/src.txt", f"{aeb_en}/{split}/tgt.txt"
        data = get_pairs(src_f, tgt_f)
        data = dedupe_data(data)
        test2_data += data

    
    test1, test2 = remove_overlap(test2_data, test1_data, val_data)
    test = test1 + test2

    subprocess.call(['mkdir', '-p', f"{combined}/val/aeb_en"])
    with open(f"{combined}/val/aeb_en/src.txt", "w") as out_src, \
         open(f"{combined}/val/aeb_en/tgt.txt", "w") as out_tgt:
        for src_line, tgt_line in val_data:
            if len(src_line.strip().split(' ')) > 3 and len(tgt_line.strip().split(' ')) > 3:
                out_src.write(src_line + '\n')
                out_tgt.write(tgt_line + '\n')

    subprocess.call(['mkdir', '-p', f"{combined}/test/aeb_en"])
    with open(f"{combined}/test/aeb_en/src.txt", "w") as out_src, \
         open(f"{combined}/test/aeb_en/tgt.txt", "w") as out_tgt:
        for src_line, tgt_line in test:
            if len(src_line.strip().split(' ')) > 3 and len(tgt_line.strip().split(' ')) > 3:
                out_src.write(src_line + '\n')
                out_tgt.write(tgt_line + '\n')

    return

### Overlap Testing ###
def check_overlap_tl_en():
    src_1_f = f"{ccmat}/tl_en/cleaned/src.txt"
    tgt_1_f = f"{ccmat}/tl_en/cleaned/tgt.txt"

    src_2_f = f"{ccalign}/tl_en/cleaned/src.txt"
    tgt_2_f = f"{ccalign}/tl_en/cleaned/tgt.txt"

    src_3_f = f"{nllb}/tl_en/cleaned/src.txt"
    tgt_3_f = f"{nllb}/tl_en/cleaned/tgt.txt"

    pairs = []
    before = 0

    dataset1 = get_pairs(src_1_f, tgt_1_f)
    print(f"Dataset 1: {len(dataset1)}")
    before += len(dataset1)
    dataset1 = dedupe_data(dataset1)
    print(f"Dataset 1 Deduped: {len(dataset1)}")

    dataset2 = get_pairs(src_2_f, tgt_2_f)
    print(f"Dataset 2: {len(dataset2)}")
    before += len(dataset2)
    dataset2 = dedupe_data(dataset2)
    print(f"Dataset 2 Deduped: {len(dataset2)}")

    dataset3 = get_pairs(src_3_f, tgt_3_f)
    print(f"Dataset 3: {len(dataset3)}")
    before += len(dataset3)
    dataset3 = dedupe_data(dataset3)
    print(f"Dataset 3 Deduped: {len(dataset3)}")

    pairs += dataset1
    pairs += dataset2
    pairs += dataset3

    print(f"All: {len(pairs)}")
    pairs = dedupe_data(pairs)
    print(f"All Deduped: {len(pairs)}")
    print(f"Duplicates: {before - len(pairs)}")


def check_overlap_mt_en():
    src_1_f = f"{dgt}/mt_en/cleaned/src.txt"
    tgt_1_f = f"{dgt}/mt_en/cleaned/tgt.txt"

    src_2_f = f"{hplt}/mt_en/cleaned/src.txt"
    tgt_2_f = f"{hplt}/mt_en/cleaned/tgt.txt"

    src_3_f = f"{nllb}/mt_en/cleaned/src.txt"
    tgt_3_f = f"{nllb}/mt_en/cleaned/tgt.txt"

    pairs = []
    before = 0

    dataset1 = get_pairs(src_1_f, tgt_1_f)
    print(f"Dataset 1: {len(dataset1)}")
    before += len(dataset1)
    dataset1 = dedupe_data(dataset1)
    print(f"Dataset 1 Deduped: {len(dataset1)}")

    dataset2 = get_pairs(src_2_f, tgt_2_f)
    print(f"Dataset 2: {len(dataset2)}")
    before += len(dataset2)
    dataset2 = dedupe_data(dataset2)
    print(f"Dataset 2 Deduped: {len(dataset2)}")

    dataset3 = get_pairs(src_3_f, tgt_3_f)
    print(f"Dataset 3: {len(dataset3)}")
    before += len(dataset3)
    dataset3 = dedupe_data(dataset3)
    print(f"Dataset 3 Deduped: {len(dataset3)}")

    pairs += dataset1
    pairs += dataset2
    pairs += dataset3

    print(f"All: {len(pairs)}")
    pairs = dedupe_data(pairs)
    print(f"All Deduped: {len(pairs)}")
    print(f"Duplicates: {before - len(pairs)}")

def get_jobs_build_subsets(args):
    JOBS = set()
    for arg in args.language_scenario:
        if arg == 'es/pt-en':
            if not check_lang_pair("es", "en"):
                es_en_src, es_en_tgt = (f"{ccmat}/es_en/cleaned/src.txt"), (f"{ccmat}/es_en/cleaned/tgt.txt")
                JOBS.add((es_en_src, es_en_tgt, f"{flores}/dev/spa_Latn.txt", f"{flores}/dev/eng_Latn.txt", f"{flores}/devtest/spa_Latn.txt", f"{flores}/devtest/eng_Latn.txt", "es", "en"))

            pt_en_src, pt_en_tgt = (f"{ccmat}/pt_en/cleaned/src.txt"), (f"{ccmat}/pt_en/cleaned/tgt.txt")
            es_pt_src, es_pt_tgt = (f"{ccmat}/es_pt/cleaned/src.txt"), (f"{ccmat}/es_pt/cleaned/tgt.txt")

            JOBS.add((pt_en_src, pt_en_tgt, f"{flores}/dev/por_Latn.txt", f"{flores}/dev/eng_Latn.txt", f"{flores}/devtest/por_Latn.txt", f"{flores}/devtest/eng_Latn.txt", "pt", "en"))
            JOBS.add((es_pt_src, es_pt_tgt, f"{flores}/dev/spa_Latn.txt", f"{flores}/dev/por_Latn.txt", f"{flores}/devtest/spa_Latn.txt", f"{flores}/devtest/por_Latn.txt", "es", "pt"))

        elif arg == 'es/an-en':
            if not check_lang_pair("es", "en"):
                es_en_src, es_en_tgt = (f"{ccmat}/es_en/cleaned/src.txt"), (f"{ccmat}/es_en/cleaned/tgt.txt")
                JOBS.add((es_en_src, es_en_tgt, f"{flores}/dev/spa_Latn.txt", f"{flores}/dev/eng_Latn.txt", f"{flores}/devtest/spa_Latn.txt", f"{flores}/devtest/eng_Latn.txt", "es", "en"))
            
            an_en_src, an_en_tgt = (f"{wikimat}/an_en/cleaned/src.txt"), (f"{wikimat}/an_en/cleaned/tgt.txt")
            es_an_src, es_an_tgt = (f"{wikimat}/es_an/cleaned/src.txt"), (f"{wikimat}/es_an/cleaned/tgt.txt")

            JOBS.add((an_en_src, an_en_tgt, f"{flores}/dev/arg_Latn.txt", f"{flores}/dev/eng_Latn.txt", f"{flores}/devtest/arg_Latn.txt", f"{flores}/devtest/eng_Latn.txt", "an", "en"))
            JOBS.add((es_an_src, es_an_tgt, f"{flores}/dev/spa_Latn.txt", f"{flores}/dev/arg_Latn.txt", f"{flores}/devtest/spa_Latn.txt", f"{flores}/devtest/arg_Latn.txt", "es", "an"))

        elif arg == "fr/mfe-en":
            if not check_lang_pair("fr", "en"):
                fr_en_src, fr_en_tgt = (f"{ccmat}/fr_en/cleaned/src.txt"), (f"{ccmat}/fr_en/cleaned/tgt.txt")
                JOBS.add((fr_en_src, fr_en_tgt, f"{flores}/dev/fra_Latn.txt", f"{flores}/dev/eng_Latn.txt", f"{flores}/devtest/fra_Latn.txt", f"{flores}/devtest/eng_Latn.txt", "fr", "en"))

            fr_mfe_src, fr_mfe_tgt = (f"{kreolmorisienmt}/fr_mfe/train/fr_mfe-fr.txt"), (f"{kreolmorisienmt}/fr_mfe/train/fr_mfe-mfe.txt")
            mfe_en_src, mfe_en_tgt = (f"{kreolmorisienmt}/mfe_en/train/mfe_en-mfe.txt"), (f"{kreolmorisienmt}/mfe_en/train/mfe_en-en.txt")

            JOBS.add((fr_mfe_src, fr_mfe_tgt, f"{kreolmorisienmt}/fr_mfe/validation/fr_mfe-fr.txt", f"{kreolmorisienmt}/fr_mfe/validation/fr_mfe-mfe.txt", f"{kreolmorisienmt}/fr_mfe/test/fr_mfe-fr.txt", f"{kreolmorisienmt}/fr_mfe/test/fr_mfe-mfe.txt", "fr", "mfe"))
            JOBS.add((mfe_en_src, mfe_en_tgt, f"{kreolmorisienmt}/mfe_en/validation/mfe_en-mfe.txt", f"{kreolmorisienmt}/mfe_en/validation/mfe_en-en.txt", f"{kreolmorisienmt}/mfe_en/test/mfe_en-mfe.txt", f"{kreolmorisienmt}/mfe_en/test/mfe_en-en.txt", "mfe", "en"))

        elif arg == "uz/kaa-en":
            ### kaa_en val ###
            if not check_subset(oldi, "kaa", "en", "val"):
                get_subset(f"{oldi}/kaa_en", "kaa", "en", "val", 2000)

            uz_en_src, uz_en_tgt = (f"{nllb}/uz_en/cleaned/src.txt"), (f"{nllb}/uz_en/cleaned/tgt.txt")
            kaa_en_src, kaa_en_tgt = (f"{oldi}/kaa_en/cleaned/src.txt"), (f"{oldi}/kaa_en/cleaned/tgt.txt")
            uz_kaa_src, uz_kaa_tgt = (f"{oldi}/uz_kaa/cleaned/src.txt"), (f"{oldi}/uz_kaa/cleaned/tgt.txt")

            JOBS.add((uz_en_src, uz_en_tgt, f"{flores}/dev/uzn_Latn.txt", f"{flores}/dev/eng_Latn.txt", f"{flores}/devtest/uzn_Latn.txt", f"{flores}/devtest/eng_Latn.txt", "uz", "en"))
            JOBS.add((kaa_en_src, kaa_en_tgt, f"{oldi}/kaa_en/val/src.txt", f"{oldi}/kaa_en/val/tgt.txt", f"{flores}/devtest/kaa_Latn.txt", f"{flores}/devtest/eng_Latn.txt", "kaa", "en"))
            JOBS.add((uz_kaa_src, uz_kaa_tgt, None, None, f"{flores}/devtest/uzn_Latn.txt", f"{flores}/devtest/kaa_Latn.txt", "uz", "kaa"))
            
        elif arg == "cs/hsb-de":
            cs_de_src, cs_de_tgt = (f"{ccmat}/cs_de/cleaned/src.txt"), (f"{ccmat}/cs_de/cleaned/tgt.txt")
            hsb_de_src, hsb_de_tgt = (f"{wmt}/hsb_de/cleaned/src.txt"), (f"{wmt}/hsb_de/cleaned/tgt.txt")

            JOBS.add((cs_de_src, cs_de_tgt, f"{flores}/dev/ces_Latn.txt", f"{flores}/dev/deu_Latn.txt", f"{flores}/devtest/ces_Latn.txt", f"{flores}/devtest/deu_Latn.txt", "cs", "de"))
            JOBS.add((hsb_de_src, hsb_de_tgt, f"{wmt}/hsb_de/devtest/devel.hsb-de.hsb", f"{wmt}/hsb_de/devtest/devel.hsb-de.de", f"{wmt}/hsb_de/devtest/devel_test.hsb-de.hsb", f"{wmt}/hsb_de/devtest/devel_test.hsb-de.de", "hsb", "de"))

        elif arg == "am/ti-en":
            am_en_src, am_en_tgt = (f"{nllb}/am_en/cleaned/src.txt"), (f"{nllb}/am_en/cleaned/tgt.txt")
            ti_en_src, ti_en_tgt = (f"{nllb}/ti_en/cleaned/src.txt"), (f"{nllb}/ti_en/cleaned/tgt.txt")
            am_ti_src, am_ti_tgt = (f"{nllb}/am_ti/cleaned/src.txt"), (f"{nllb}/am_ti/cleaned/tgt.txt")

            JOBS.add((am_en_src, am_en_tgt, f"{flores}/dev/amh_Ethi.txt", f"{flores}/dev/eng_Latn.txt", f"{flores}/devtest/amh_Ethi.txt", f"{flores}/devtest/eng_Latn.txt", "am", "en"))
            JOBS.add((ti_en_src, ti_en_tgt, f"{flores}/dev/tir_Ethi.txt", f"{flores}/dev/eng_Latn.txt", f"{flores}/devtest/tir_Ethi.txt", f"{flores}/devtest/eng_Latn.txt", "ti", "en"))
            JOBS.add((am_ti_src, am_ti_tgt, f"{flores}/dev/amh_Ethi.txt", f"{flores}/dev/tir_Ethi.txt", f"{flores}/devtest/amh_Ethi.txt", f"{flores}/devtest/tir_Ethi.txt", "am", "ti"))

        elif arg == "tl/bik-en":
            ### bik_en val and test ###
            if not check_subset(wikimed, "bik", "en", "val"):
                get_subset(f"{wikimed}/bik_en", "bik", "en", "val", 1000)
            if not check_subset(wikimed, "bik", "en", "test"):
                get_subset(f"{wikimed}/bik_en", "bik", "en", "test", 1000)

            tl_en_src, tl_en_tgt = (f"{nllb}/tl_en/cleaned/src.txt"), (f"{nllb}/tl_en/cleaned/tgt.txt")
            bik_en_src, bik_en_tgt = (f"{wikimed}/bik_en/cleaned/src.txt"), (f"{wikimed}/bik_en/cleaned/tgt.txt")

            JOBS.add((tl_en_src, tl_en_tgt, f"{flores}/dev/fil_Latn.txt", f"{flores}/dev/eng_Latn.txt", f"{flores}/devtest/fil_Latn.txt", f"{flores}/devtest/eng_Latn.txt", "tl", "en"))
            JOBS.add((bik_en_src, bik_en_tgt, f"{wikimed}/bik_en/val/src.txt", f"{wikimed}/bik_en/val/tgt.txt", f"{wikimed}/bik_en/test/src.txt", f"{wikimed}/bik_en/test/tgt.txt", "bik", "en"))

            # dataset comparison
            # combine_datasets(ccmat, ccalign, nllb, 'tl', 'en')

            # tlx_enx_src, tlx_enx_tgt = (f"{nllb}/tl_en/cleaned/src.txt"), (f"{nllb}/tl_en/cleaned/tgt.txt")
            # tly_eny_src, tly_eny_tgt = (f"{combined}/tl_en/src.txt"), (f"{combined}/tl_en/tgt.txt")

            # JOBS.add((tlx_enx_src, tlx_enx_tgt, f"{flores}/dev/fil_Latn.txt", f"{flores}/dev/eng_Latn.txt", f"{flores}/devtest/fil_Latn.txt", f"{flores}/devtest/eng_Latn.txt", "tlx", "enx"))
            # JOBS.add((tly_eny_src, tly_eny_tgt, f"{flores}/dev/fil_Latn.txt", f"{flores}/dev/eng_Latn.txt", f"{flores}/devtest/fil_Latn.txt", f"{flores}/devtest/eng_Latn.txt", "tly", "eny"))

        elif arg == "bn/rhg-en":
            ### rhg_en val and test ###
            if not check_subset(twb, "rhg", "en", "val"):
                get_subset(f"{twb}/rhg_en", "rhg", "en", "val", 250)
            if not check_subset(twb, "rhg", "en", "test"):
                get_subset(f"{twb}/rhg_en", "rhg", "en", "test", 250)

            bn_en_src, bn_en_tgt = (f"{ccmat}/bn_en/cleaned/src.txt"), (f"{ccmat}/bn_en/cleaned/tgt.txt")
            rhg_en_src, rhg_en_tgt = (f"{twb}/rhg_en/cleaned/src.txt"), (f"{twb}/rhg_en/cleaned/tgt.txt")

            JOBS.add((bn_en_src, bn_en_tgt, f"{flores}/dev/ben_Beng.txt", f"{flores}/dev/eng_Latn.txt", f"{flores}/devtest/ben_Beng.txt", f"{flores}/devtest/eng_Latn.txt", "bn", "en"))
            JOBS.add((rhg_en_src, rhg_en_tgt, f"{twb}/rhg_en/val/src.txt", f"{twb}/rhg_en/val/tgt.txt", f"{twb}/rhg_en/test/src.txt", f"{twb}/rhg_en/test/tgt.txt", "rhg", "en"))

        elif arg == "mt/aeb-en":
            if not check_lang_pair("mt", "en"):
                mt_en_src, mt_en_tgt = (f"{nllb}/mt_en/cleaned/src.txt"), (f"{nllb}/mt_en/cleaned/tgt.txt")
                JOBS.add((mt_en_src, mt_en_tgt, f"{flores}/dev/mlt_Latn.txt", f"{flores}/dev/eng_Latn.txt", f"{flores}/devtest/mlt_Latn.txt", f"{flores}/devtest/eng_Latn.txt", "mt", "en"))

            combine_test_val_sets_aeb_en()
            aeb_en_src, aeb_en_tgt = (f"{ldc}/aeb_en/cleaned/src.txt"), (f"{ldc}/aeb_en/cleaned/tgt.txt")
            JOBS.add((aeb_en_src, aeb_en_tgt, f"{combined}/val/aeb_en/src.txt", f"{combined}/val/aeb_en/tgt.txt", f"{combined}/test/aeb_en/src.txt", f"{combined}/test/aeb_en/tgt.txt", "aeb", "en"))

            ### dataset comparison
            # combine_datasets(dgt, hplt, nllb, 'mt', 'en')

            # mtx_enx_src, mtx_enx_tgt = (f"{nllb}/mt_en/cleaned/src.txt"), (f"{nllb}/mt_en/cleaned/tgt.txt")
            # mty_eny_src, mty_eny_tgt = (f"{combined}/mt_en/src.txt"), (f"{combined}/mt_en/tgt.txt")

            # JOBS.add((mtx_enx_src, mtx_enx_tgt, f"{flores}/dev/mlt_Latn.txt", f"{flores}/dev/eng_Latn.txt", f"{flores}/devtest/mlt_Latn.txt", f"{flores}/devtest/eng_Latn.txt", "mtx", "enx"))
            # JOBS.add((mty_eny_src, mty_eny_tgt, f"{flores}/dev/mlt_Latn.txt", f"{flores}/dev/eng_Latn.txt", f"{flores}/devtest/mlt_Latn.txt", f"{flores}/devtest/eng_Latn.txt", "mty", "eny"))
        
        elif arg == "mt/ary-en":
            if not check_lang_pair("mt", "en"):
                mt_en_src, mt_en_tgt = (f"{nllb}/mt_en/cleaned/src.txt"), (f"{nllb}/mt_en/cleaned/tgt.txt")
                JOBS.add((mt_en_src, mt_en_tgt, f"{flores}/dev/mlt_Latn.txt", f"{flores}/dev/eng_Latn.txt", f"{flores}/devtest/mlt_Latn.txt", f"{flores}/devtest/eng_Latn.txt", "mt", "en"))
            
            ary_en_src, ary_en_tgt = (f"{doda}/ary_en/cleaned/src.txt", f"{doda}/ary_en/cleaned/tgt.txt")
            JOBS.add((ary_en_src, ary_en_tgt, f"{flores}/dev/ary_Arab.txt", f"{flores}/dev/eng_Latn.txt", f"{flores}/devtest/ary_Arab.txt", f"{flores}/devtest/eng_Latn.txt", "ary", "en"))

            pass

        elif arg == "fr/crs-en":
            if not check_lang_pair("fr", "en"):
                fr_en_src, fr_en_tgt = (f"{ccmat}/fr_en/cleaned/src.txt"), (f"{ccmat}/fr_en/cleaned/tgt.txt")
                JOBS.add((fr_en_src, fr_en_tgt, f"{flores}/dev/fra_Latn.txt", f"{flores}/dev/eng_Latn.txt", f"{flores}/devtest/fra_Latn.txt", f"{flores}/devtest/eng_Latn.txt", "fr", "en"))
            ### crs val and test ###
            if not check_subset(mt560, "crs", "en", "val"):
                get_subset(f"{mt560}/crs_en", "crs", "en", "val", 2000)
            if not check_subset(mt560, "crs", "en", "test"):
                get_subset(f"{mt560}/crs_en", "crs", "en", "test", 2000)
            
            crs_en_src, crs_en_tgt = (f"{mt560}/crs_en/cleaned/src.txt"), (f"{mt560}/crs_en/cleaned/tgt.txt")
            JOBS.add((crs_en_src, crs_en_tgt, f"{mt560}/crs_en/val/src.txt", f"{mt560}/crs_en/val/tgt.txt", f"{mt560}/crs_en/test/src.txt", f"{mt560}/crs_en/test/tgt.txt", "crs", "en"))

        elif arg == "ca/oc-en":
            ca_en_src, ca_en_tgt = (f"{nllb}/ca_en/cleaned/src.txt"), (f"{nllb}/ca_en/cleaned/tgt.txt")
            oc_en_src, oc_en_tgt = (f"{nllb}/oc_en/cleaned/src.txt"), (f"{nllb}/oc_en/cleaned/tgt.txt")

            JOBS.add((ca_en_src, ca_en_tgt, f"{flores}/dev/cat_Latn.txt", f"{flores}/dev/eng_Latn.txt", f"{flores}/devtest/cat_Latn.txt", f"{flores}/devtest/eng_Latn.txt", "ca", "en"))
            JOBS.add((oc_en_src, oc_en_tgt, f"{flores}/dev/oci_Latn.txt", f"{flores}/dev/eng_Latn.txt", f"{flores}/devtest/oci_Latn.txt", f"{flores}/devtest/eng_Latn.txt", "oc", "en"))

        elif arg == "es/cbk-en":
            pass

    return JOBS


if __name__ == "__main__":
    args = get_args()
    JOBS = get_jobs_build_subsets(args)

    for job in JOBS:
        src_train, tgt_train, src_val, tgt_val, src_test, tgt_test, src_lang, tgt_lang = job
        main(src_train, tgt_train, src_val, tgt_val, src_test, tgt_test, src_lang, tgt_lang)

    # check_overlap_tl_en()
    # check_overlap_mt_en()


