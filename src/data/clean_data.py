import subprocess
import os
from pathlib import Path
from aksharamukha import transliterate
from tqdm import tqdm
import argparse
from utilities.utilities import set_env

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

os.chdir("src/data/data-cleaning-pipeline")

def run_cleaning_job(job):
    root_dir, sub_dir, src_lang, tgt_lang, src_f, tgt_f = job

    save_dir = Path(DATA_HOME) / "raw" / root_dir / sub_dir

    src_file = save_dir / f"{sub_dir}-{src_f}.txt"
    tgt_file = save_dir / f"{sub_dir}-{tgt_f}.txt"

    print(f"---------------- cleaning {sub_dir} ----------------")

    cmd = [
        "python3", "pipeline.py",
        "-t", str(save_dir),
        "-srclang", src_lang,
        "-tgtlang", tgt_lang,
        "-srcpath", str(src_file),
        "-tgtpath", str(tgt_file),
        "-d", "-v"
    ]

    result = subprocess.run(cmd, capture_output=False, text=True)
    
    if result.returncode != 0:
        print(f"FAILED: {sub_dir}\n{result.stderr}")
    else:
        print(f"SUCCESS: {sub_dir}")


def romanize_bengali():
    # flores dev
    with open(f"{DATA_HOME}/raw/flores+/dev/ben_Beng.txt", "r") as infile:
        with open(f"{DATA_HOME}/raw/flores+/dev/ben_Latn.txt", "w") as outfile:
            for line in infile:
                outfile.write(transliterate.process("Bengali", "RomanColloquial", line))
    
    # flores devtest
    with open(f"{DATA_HOME}/raw/flores+/devtest/ben_Beng.txt", "r") as infile:
        with open(f"{DATA_HOME}/raw/flores+/devtest/ben_Latn.txt", "w") as outfile:
            for line in infile:
                outfile.write(transliterate.process("Bengali", "RomanColloquial", line))

    # CCMatrix
    with open(f"{DATA_HOME}/raw/CCMatrix/bn_en/cleaned/src.txt", "r") as infile, \
         open(f"{DATA_HOME}/raw/CCMatrix/bn_en/bn_en-bn_Latn.txt", "w") as outfile:
            
        with tqdm(unit="lines", desc="Transliterating Bengali") as pbar:
            for line in infile:
                outfile.write(transliterate.process("Bengali", "RomanColloquial", line))
                pbar.update(1)

def check_lang_pair(dataset, src, tgt):
    return os.path.exists(f"{dataset}/{src}_{tgt}/cleaned")

def get_args():
    parser = argparse.ArgumentParser(description="Clean Language Scenarios")

    SCENARIOS = ['es/pt-en', 'es/an-en', 'fr/mfe-en', 'uz/kaa-en', 'cs/hsb-de', 
                'am/ti-en', 'tl/bik-en', 'bn/rhg-en', 'mt/aeb-en', 'mt/ary-en', 
                'fr/crs-en', 'ca/oc-en', 'es/cbk-en']

    parser.add_argument('--language_scenario', '-l', type=str, nargs='+', 
                        choices=SCENARIOS, default=SCENARIOS)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    JOBS = set()
    rom_ben = False
    rom_tun = False
    for arg in args.language_scenario:
        if arg == 'es/pt-en':
            if not check_lang_pair(ccmat, "es", "en"):
                JOBS.add(("CCMatrix", "es_en", "es-ES", "en-US", "es", "en"))
            JOBS.add(("CCMatrix", "pt_en", "pt-PT", "en-US", "pt", "en"))
            JOBS.add(("CCMatrix", "es_pt", "es-ES", "pt-PT", "es", "pt"))

        elif arg == 'es/an-en':
            if not check_lang_pair(ccmat, "es", "en"):
                JOBS.add(("CCMatrix", "es_en", "es-ES", "en-US", "es", "en"))
            JOBS.add(("WikiMatrix", "an_en", "an-ES", "en-US", "an", "en"))
            JOBS.add(("WikiMatrix", "es_an", "es-ES", "an-ES", "es", "an"))

        elif arg == "fr/mfe-en":
            if not check_lang_pair(ccmat, "fr", "en"):
                JOBS.add(("CCMatrix", "fr_en", "fr-FR", "en-US", "fr", "en"))

        elif arg == "uz/kaa-en":
            JOBS.add(("NLLB", "uz_en", "uz-UZ", "en-US", "uz", "en"))
            JOBS.add(("OLDI", "kaa_en", "kaa-UZ", "en-US", "kaa", "en"))
            JOBS.add(("OLDI", "uz_kaa", "uz-UZ", "kaa-UZ", "uz", "kaa"))
            
        elif arg == "cs/hsb-de":
            JOBS.add(("CCMatrix", "cs_de", "cs-CZ", "de-DE", "cs", "de"))
            JOBS.add(("WMT20", "hsb_de", "hsb-CZ", "de-DE", "hsb", "de"))

        elif arg == "am/ti-en":
            subprocess.call(["cp", "config/am-ET.yaml", "config/ti-ET.yaml"])
            JOBS.add(("NLLB", "am_en", "am-ET", "en-US", "am", "en"))
            JOBS.add(("NLLB", "ti_en", "ti-ET", "en-US", "ti", "en"))
            JOBS.add(("NLLB", "am_ti", "am-ET", "ti-ET", "am", "ti"))

        elif arg == "tl/bik-en":
            JOBS.add(("NLLB", "tl_en", "tl-PH", "en-US", "tl", "en"))
            # JOBS.add(("CCMatrix", "tl_en", "tl-PH", "en-US", "tl", "en"))
            # JOBS.add(("CCAligned", "tl_en", "tl-PH", "en-US", "tl", "en"))
            JOBS.add(("wikimedia", "bik_en", "bik-PH", "en-US", "bik", "en"))

        elif arg == "bn/rhg-en":
            JOBS.add(("CCMatrix", "bn_en", "bn-BD", "en-US", "bn", "en"))
            JOBS.add(("TWB", "rhg_en", "rhg-BD", "en-US", "rhg", "en"))
            rom_ben = True

        elif arg == "mt/aeb-en":
            if not check_lang_pair(nllb, "mt", "en"):
                JOBS.add(("NLLB", "mt_en", "mt-MT", "en-US", "mt", "en"))
                # JOBS.add(("DGT", "mt_en", "mt-MT", "en-US", "mt", "en"))
                # JOBS.add(("HPLT", "mt_en", "mt-MT", "en-US", "mt", "en"))
            JOBS.add(("LDC", "aeb_en", "aeb-TN", "en-US", "aeb", "en"))

        elif arg == "mt/ary-en":
            if not check_lang_pair(nllb, "mt", "en"):
                JOBS.add(("NLLB", "mt_en", "mt-MT", "en-US", "mt", "en"))
                # JOBS.add(("DGT", "mt_en", "mt-MT", "en-US", "mt", "en"))
                # JOBS.add(("HPLT", "mt_en", "mt-MT", "en-US", "mt", "en"))
            # JOBS.add(("DODa", "ary_en", "ary-MA", "en-US", "ary", "en")) # Which script to clean?
            rom_tun = True

        elif arg == "fr/crs-en":
            if not check_lang_pair(ccmat, "fr", "en"):
                JOBS.add(("CCMatrix", "fr_en", "fr-FR", "en-US", "fr", "en"))
            JOBS.add(("MT560", "crs_en", "crs-SC", "en-US", "crs", "en"))

        elif arg == "ca/oc-en":
            JOBS.add(("NLLB", "ca_en", "ca-ES", "en-US", "ca", "en"))
            JOBS.add(("NLLB", "oc_en", "oc-FR", "en-US", "oc", "en"))

        elif arg == "es/cbk-en":
            if not check_lang_pair(ccmat, "es", "en"):
                JOBS.add(("CCMatrix", "es_en", "es-ES", "en-US", "es", "en"))
            ## cbk-en not yet implemented



    for job in JOBS:
        run_cleaning_job(job)

    if rom_ben:
        romanize_bengali()

    # if rom_tun:
    #     romanize_tunisian()
    





