import subprocess
import os
import sys
from pathlib import Path
from aksharamukha import transliterate
from tqdm import tqdm


DATA_HOME = sys.argv[1]


JOBS = [
    # ("NLLB", "uz_en", "uz-UZ", "en-US", "uz", "en"),
    # ("OLDI", "uz_kaa", "uz-UZ", "kaa-UZ", "uz", "kaa"),
    # ("OLDI", "kaa_en", "kaa-UZ", "en-US", "kaa", "en"),
    # ("NLLB", "am_en", "am-ET", "en-US", "am", "en"),
    # ("NLLB", "am_ti", "am-ET", "ti-ET", "am", "ti"),
    # ("NLLB", "ti_en", "ti-ET", "en-US", "ti", "en"),
    # ("NLLB", "tl_en", "tl-PH", "en-US", "tl", "en"),
    # ("CCMatrix", "tl_en", "tl-PH", "en-US", "tl", "en"),
    # ("CCAligned", "tl_en", "tl-PH", "en-US", "tl", "en"),
    # ("wikimedia", "bik_en", "bik-PH", "en-US", "bik", "en"),
    # ("CCMatrix", "es_en", "es-ES", "en-US", "es", "en"),
    # ("CCMatrix", "es_pt", "es-ES", "pt-PT", "es", "pt"),
    # ("CCMatrix", "pt_en", "pt-PT", "en-US", "pt", "en"),
    # ("WikiMatrix", "an_en", "an-ES", "en-US", "an", "en"),
    # ("WikiMatrix", "es_an", "es-ES", "an-ES", "es", "an"),
    # ("CCMatrix", "fr_en", "fr-FR", "en-US", "fr", "en"),
    # ("CCMatrix", "cs_de", "cs-CZ", "de-DE", "cs", "de"),
    # ("WMT20", "hsb_de", "hsb-CZ", "de-DE", "hsb", "de"),
    ("CCMatrix", "bn_en", "bn-BD", "en-US", "bn", "en")
    # ("TWB", "rhg_en", "rhg-BD", "en-US", "rhg", "en"),
    # ("MT560", "crs_en", "crs-SC", "en-US", "crs", "en"),
    # ("NLLB", "ca_en", "ca-ES", "en-US", "ca", "en"),
    # ("NLLB", "oc_en", "oc-FR", "en-US", "oc", "en"),
    # ("NLLB", "mt_en", "mt-MT", "en-US", "mt", "en"),
    # ("DGT", "mt_en", "mt-MT", "en-US", "mt", "en"),
    # ("HPLT", "mt_en", "mt-MT", "en-US", "mt", "en"),
    # ("DODa", "ary_en", "ary-MA", "en-US", "ary", "en")

]

def run_cleaning_job(job):
    root_dir, sub_dir, src_lang, tgt_lang, src_f, tgt_f = job

    save_dir = Path(DATA_HOME) / "raw" / root_dir / sub_dir

    src_file = save_dir / f"{sub_dir}-{src_f}.txt"
    tgt_file = save_dir / f"{sub_dir}-{tgt_f}.txt"

    # subprocess.call(["mkdir", "-p", save_dir])

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
    # with open(f"{DATA_HOME}/raw/flores+/dev/ben_Beng.txt", "r") as infile:
    #     with open(f"{DATA_HOME}/raw/flores+/dev/ben_Latn.txt", "w") as outfile:
    #         for line in infile:
    #             outfile.write(transliterate.process("Bengali", "RomanReadable", line))
    
    # # flores devtest
    # with open(f"{DATA_HOME}/raw/flores+/devtest/ben_Beng.txt", "r") as infile:
    #     with open(f"{DATA_HOME}/raw/flores+/devtest/ben_Latn.txt", "w") as outfile:
    #         for line in infile:
    #             outfile.write(transliterate.process("Bengali", "RomanReadable", line))

    # CCMatrix

    with open(f"{DATA_HOME}/raw/CCMatrix/bn_en/bn_en-bn.txt", "r") as infile, \
         open(f"{DATA_HOME}/raw/CCMatrix/bn_en/bn_en-bn-Latn.txt", "w") as outfile:
            
        with tqdm(total=os.path.getsize(f"{DATA_HOME}/raw/CCMatrix/bn_en/bn_en-bn.txt"), unit="B", unit_scale=True, desc="Transliterating Bengali") as pbar:
            for line in infile:
                outfile.write(transliterate.process("Bengali", "RomanReadable", line))
                pbar.update(len(line.encode('utf-8')))

if __name__ == "__main__":
    for job in JOBS:
        run_cleaning_job(job)


    ## romanize Bengali
    # romanize_bengali()
    





