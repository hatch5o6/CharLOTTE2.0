from datasets import load_dataset
import huggingface_hub
import pandas as pd
import subprocess
from lxml import etree
import os
import sys
import requests
from tqdm import tqdm

# filepaths
DATA_HOME = sys.argv[1]

HF_TOKEN = sys.argv[2]

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
twb=f"{raw_data}/TWB"
chavmt=f"{raw_data}/ChavacanoMT"
dgt=f"{raw_data}/DGT"
hplt=f"{raw_data}/HPLT"
doda=f"{raw_data}/DODa"
flores=f"{raw_data}/flores+"


def process_nllb(src_key, tgt_key, src, tgt, k=1000, switch=False):
    lang_pair = f"{src}_{tgt}"
    subprocess.call(["rm", "-rf", f"{nllb}/{lang_pair}"])
    subprocess.call(["mkdir", "-p", f"{nllb}/{lang_pair}"])
    lang_pair_path = f"{nllb}/{lang_pair}/{lang_pair}"

    dataset = load_dataset("allenai/nllb", f"{src_key}-{tgt_key}", streaming=True)

    with open(f"{lang_pair_path}-{src}.txt", "w", encoding="utf-8") as f_src, \
         open(f"{lang_pair_path}-{tgt}.txt", "w", encoding="utf-8") as f_tgt:

        for row in tqdm(dataset["train"].take(k), total=k, desc=f"Downloading {src_key}-{tgt_key} dataset"):
            trans = row["translation"]

            if switch == False:
                src_text = trans[src_key].replace("\n", " ")
                tgt_text = trans[tgt_key].replace("\n", " ")

            else:
                src_text = trans[tgt_key].replace("\n", " ")
                tgt_text = trans[src_key].replace("\n", " ")

            f_src.write(src_text + "\n")
            f_tgt.write(tgt_text + "\n")
    
    print(f"{k} lines downloaded successfully")

def process_ccmatrix_subset(mb, src, tgt, url, trunc=True):
    size = mb * 1024 * 1024
    lang_pair = f"{src}_{tgt}"
    subprocess.call(["rm", "-rf", f"{ccmat}/{lang_pair}"])
    subprocess.call(["mkdir", "-p", f"{ccmat}/{lang_pair}"])
    lang_pair_path = f"{ccmat}/{lang_pair}/{lang_pair}"
    tmx_path = f"{lang_pair_path}.tmx"

    download_tmx_subset(url, tmx_path, size)
    if trunc==True:
        trunc_tmx(tmx_path)
    # sed_command = f"sed -i 's/[\\x00-\\x08\\x0B\\x0C\\x0E-\\x1F\\x7F]//g' {tmx_path}" # remove bad bytes
    sed_command = (
    f"sed -i 's/[\\x00-\\x08\\x0B\\x0C\\x0E-\\x1F\\x7F]//g; "
    f"s/\\xEF\\xBF\\xBE//g; "   # UTF-8 encoding of U+FFFE
    f"s/\\xEF\\xBF\\xBF//g' "   # UTF-8 encoding of U+FFFF
    f"{tmx_path}"
)
    subprocess.run(sed_command, shell=True, check=True)
    extract_tmx_to_moses(tmx_path, src, tgt, f"{lang_pair_path}-{src}.txt", f"{lang_pair_path}-{tgt}.txt")

    subprocess.call(["rm", tmx_path])


################################## uz / kaa <--> en ##################################
def kaa_uz_en_oldi():
    # load dataset
    dilmash_corpus = load_dataset("tahrirchi/dilmash")
  
    df_kaa_uzb = pd.DataFrame(dilmash_corpus["kaa_uzb"])
    df_kaa_eng = pd.DataFrame(dilmash_corpus["kaa_eng"])

    # convert to text files
    subprocess.call(["rm", "-rf", f"{oldi}/uz_kaa"])
    subprocess.call(["rm", "-rf", f"{oldi}/kaa_en"])

    subprocess.call(["mkdir", "-p", f"{oldi}/uz_kaa"])
    subprocess.call(["mkdir", "-p", f"{oldi}/kaa_en"])
    uz_kaa = f"{oldi}/uz_kaa/uz_kaa"
    kaa_en = f"{oldi}/kaa_en/kaa_en"

    df_kaa_uzb["src_sent"].to_csv(f"{uz_kaa}-kaa.txt", index=False, header=False)
    df_kaa_uzb["tgt_sent"].to_csv(f"{uz_kaa}-uz.txt", index=False, header=False)
    
    df_kaa_eng["src_sent"].to_csv(f"{kaa_en}-kaa.txt", index=False, header=False)
    df_kaa_eng["tgt_sent"].to_csv(f"{kaa_en}-en.txt", index=False, header=False)


################################## tl / bik <--> en ##################################
def tl_en_ccalign():
    subprocess.call(["rm", "-rf", f"{ccalign}"])
    subprocess.call(["mkdir", "-p", f"{ccalign}/tl_en"])

    tl_en = f"{ccalign}/tl_en/tl_en"
    tmx_path = f"{tl_en}.tmx"

    subprocess.call(["wget", "https://object.pouta.csc.fi/OPUS-CCAligned/v1/tmx/en-tl.tmx.gz", "-O", tmx_path + ".gz"])

    subprocess.call(["gunzip", tmx_path + ".gz"])

    sed_command = (
    f"sed -i 's/[\\x00-\\x08\\x0B\\x0C\\x0E-\\x1F\\x7F]//g; "
    f"s/\\xEF\\xBF\\xBE//g; "   # UTF-8 encoding of U+FFFE
    f"s/\\xEF\\xBF\\xBF//g' "   # UTF-8 encoding of U+FFFF
    f"{tmx_path}")
    subprocess.run(sed_command, shell=True, check=True)

    extract_tmx_to_moses(tmx_path, "tl", "en", f"{tl_en}-tl.txt", f"{tl_en}-en.txt")

    subprocess.call(["rm", tmx_path])

def bik_en_wikimedia():
    # load dataset as text files
    subprocess.call(["rm", "-rf", f"{wikimed}/bik_en"])
    subprocess.call(["mkdir", "-p", f"{wikimed}/bik_en"])
    bik_en = f"{wikimed}/bik_en/bik_en"

    subprocess.call(["opus_read", "-s", "bcl", "-t", "en", "-d", "wikimedia", "-p", "moses", "--write",
     f"{bik_en}-bik.txt", f"{bik_en}-en.txt", "-q"])

    # delete zip file
    subprocess.call(["rm", "wikimedia_latest_moses_bcl-en.txt.zip"])


################################## es / an <--> en ##################################
# es <--> en downloaded in es / pt <--> en
def es_an_en_wikimatrix():
    # load datasets as text files
    subprocess.call(["rm", "-rf", f"{wikimat}/an_en"])
    subprocess.call(["mkdir", "-p", f"{wikimat}/an_en"])
    an_en = f"{wikimat}/an_en/an_en"
    subprocess.call(["opus_read", "-s", "an", "-t", "en", "-d", "WikiMatrix", "-p", "moses", "--write",
     f"{an_en}-an.txt", f"{an_en}-en.txt", "-q"])

    subprocess.call(["rm", "-rf", f"{wikimat}/es_an"])
    subprocess.call(["mkdir", "-p", f"{wikimat}/es_an"])
    es_an = f"{wikimat}/es_an/es_an"
    subprocess.call(["opus_read", "-s", "es", "-t", "an", "-d", "WikiMatrix", "-p", "moses", "--write",
     f"{es_an}-es.txt", f"{es_an}-an.txt", "-q"])

    # delete zip files
    subprocess.call(["rm", "WikiMatrix_latest_moses_an-en.txt.zip"])
    subprocess.call(["rm", "WikiMatrix_latest_moses_an-es.txt.zip"])


################################## fr / mfe <--> en ##################################
def mfe_en_fr_kreyolmt():
    # ## fr <--> mfe ##
    subprocess.call(["rm", "-rf", kreyolmt])
    subprocess.call(["mkdir", "-p", f"{kreyolmt}/fr_mfe"])
    dataset = load_dataset("jhu-clsp/kreyol-mt", "mfe-fra")

    for split in ["train", "test", "validation"]:
        df = pd.DataFrame(dataset[split])

        df["mfe"] = df["translation"].apply(lambda x: x["src_text"])
        df["fr"] = df["translation"].apply(lambda x: x["tgt_text"])

        subprocess.call(["mkdir", "-p", f"{kreyolmt}/fr_mfe/{split}"])
        fr_mfe = f"{kreyolmt}/fr_mfe/{split}/fr_mfe"
        df['mfe'].to_csv(f"{fr_mfe}-mfe.txt", index=False, header=False)
        df['fr'].to_csv(f"{fr_mfe}-fr.txt", index=False, header=False)
    
    ## mfe <--> en ##
    subprocess.call(["mkdir", "-p", f"{kreyolmt}/mfe_en"])
    dataset = load_dataset("jhu-clsp/kreyol-mt", "mfe-eng")

    for split in ["train", "test", "validation"]:
        df = pd.DataFrame(dataset[split])

        df["mfe"] = df["translation"].apply(lambda x: x["src_text"])
        df["en"] = df["translation"].apply(lambda x: x["tgt_text"])

        subprocess.call(["mkdir", "-p", f"{kreyolmt}/mfe_en/{split}"])
        mfe_en = f"{kreyolmt}/mfe_en/{split}/mfe_en"
        df['mfe'].to_csv(f"{mfe_en}-mfe.txt", index=False, header=False)
        df['en'].to_csv(f"{mfe_en}-en.txt", index=False, header=False)

def mfe_en_fr_kreolmorisienmt():
    ## fr <--> mfe ##
    subprocess.call(["rm", "-rf", kreolmorisienmt])
    subprocess.call(["mkdir", "-p", f"{kreolmorisienmt}/fr_mfe"])
    dataset = load_dataset("prajdabre/MorisienMT", "fr-cr")

    for split in ["train", "test", "validation"]:
        df = pd.DataFrame(dataset[split])

        subprocess.call(["mkdir", "-p", f"{kreolmorisienmt}/fr_mfe/{split}"])
        fr_mfe = f"{kreolmorisienmt}/fr_mfe/{split}/fr_mfe"
        df['target'].to_csv(f"{fr_mfe}-mfe.txt", index=False, header=False)
        df['input'].to_csv(f"{fr_mfe}-fr.txt", index=False, header=False)
    
    ## mfe <--> en ##
    subprocess.call(["mkdir", "-p", f"{kreolmorisienmt}/mfe_en"])
    dataset = load_dataset("prajdabre/MorisienMT", "en-cr")

    for split in ["train", "test", "validation"]:
        df = pd.DataFrame(dataset[split])

        subprocess.call(["mkdir", "-p", f"{kreolmorisienmt}/mfe_en/{split}"])
        mfe_en = f"{kreolmorisienmt}/mfe_en/{split}/mfe_en"
        df['target'].to_csv(f"{mfe_en}-mfe.txt", index=False, header=False)
        df['input'].to_csv(f"{mfe_en}-en.txt", index=False, header=False)

################################## fr / mfe <--> en ##################################
def crs_en_mt560():
    subprocess.call(["rm", "-rf", mt560])
    subprocess.call(["mkdir", "-p", f"{mt560}/crs_en"])

    crs_en = f"{mt560}/crs_en/crs_en"

    dataset = load_dataset("michsethowusu/english-seychelles-french-creole_sentence-pairs_mt560", streaming=True)

    with open(f"{crs_en}-crs.txt", "w", encoding="utf-8") as f_src, \
         open(f"{crs_en}-en.txt", "w", encoding="utf-8") as f_tgt:

        for row in tqdm(dataset["train"], total=188361, desc=f"Downloading crs_en dataset"):
            src_text = row["crs"].replace("\n", " ")
            tgt_text = row["eng"].replace("\n", " ")

            f_src.write(src_text + "\n")
            f_tgt.write(tgt_text + "\n")
    
    print(f"crs_en downloaded successfully")



################################## cs / hsb <--> de ##################################
def hsb_de_wmt20():
    # train sets
    subprocess.call(["rm", "-rf", f"{wmt}/hsb_de"])
    subprocess.call(["mkdir", "-p", f"{wmt}/hsb_de"])
    hsb_de = f"{wmt}/hsb_de/hsb_de"
    subprocess.call(["wget", "https://www.statmt.org/wmt20/unsup_and_very_low_res/train.hsb-de.hsb.gz"])
    subprocess.call(["gunzip", "train.hsb-de.hsb.gz"])
    subprocess.call(["mv", "train.hsb-de.hsb", f"{hsb_de}-hsb.txt"])

    subprocess.call(["wget", "https://www.statmt.org/wmt20/unsup_and_very_low_res/train.hsb-de.de.gz"])
    subprocess.call(["gunzip", "train.hsb-de.de.gz"])
    subprocess.call(["mv", "train.hsb-de.de", f"{hsb_de}-de.txt"])

    # dev and test
    subprocess.call(["mkdir", "-p", f"{wmt}/hsb_de/devtest"])
    subprocess.call(["wget", "https://www.statmt.org/wmt20/unsup_and_very_low_res/devtest.tar.gz"])
    subprocess.call(["tar", "-xzvf", "devtest.tar.gz", "-C", f"{wmt}/hsb_de/devtest/"])
    subprocess.call(["rm", "devtest.tar.gz"])


################################## bn / rhg <--> en ##################################
def rh_en_twb():
    """#### Rohingya <--> English dataset must be downloaded with a free account at
    https://gamayun.translatorswb.org/download/gamayun-5k-english-rohingya/
    and placed in data/ directory ####"""

    subprocess.call(["rm", "-rf", f"{twb}/rhg_en"])
    subprocess.call(["mkdir", "-p", f"{twb}/rhg_en"])
    rhg_en = f"{twb}/rhg_en/rhg_en"

    subprocess.call(["unzip", "Gamayun_core_kit5k_eng-rhg.zip"])
    subprocess.call(["mv", "gamayun_kit5k.eng", f"{rhg_en}-en.txt"])
    subprocess.call(["mv", "gamayun_kit5k.rhg", f"{rhg_en}-rhg.txt"])

    subprocess.call(["rm", "Gamayun_core_kit5k_eng-rhg.zip"])


################################## es / cbk <--> en ##################################
def cbk_en_chavmt():
    subprocess.call(["rm", "-rf", chavmt])
    subprocess.call(["mkdir", "-p", f"{chavmt}/cbk_en"])

    cbk_en = f"{chavmt}/cbk_en/cbk_en"

    dataset = load_dataset("", streaming=True)

    # with open(f"{cbk_en}-crs.txt", "w", encoding="utf-8") as f_src, \
    #      open(f"{cbk_en}-en.txt", "w", encoding="utf-8") as f_tgt:

    #     for row in tqdm(dataset["train"], total=188361, desc=f"Downloading cbk_en dataset"):
    #         src_text = row["crs"].replace("\n", " ")
    #         tgt_text = row["eng"].replace("\n", " ")

    #         f_src.write(src_text + "\n")
    #         f_tgt.write(tgt_text + "\n")
    
    # print(f"cbk_en downloaded successfully")

################################## mt / aeb <--> en ##################################
def mt_en_dgt():
    subprocess.call(["rm", "-rf", dgt])
    subprocess.call(["mkdir", "-p", f"{dgt}/mt_en"])
    mt_en = f"{dgt}/mt_en/mt_en"

    subprocess.call(["opus_read", "-s", "mt", "-t", "en", "-d", "DGT", "-p", "moses", "--write",
     f"{mt_en}-mt.txt", f"{mt_en}-en.txt", "-q"])

    # delete zip file
    subprocess.call(["rm", "DGT_latest_moses_en-mt.txt.zip"])

def mt_en_hplt():
    # load dataset as text files
    subprocess.call(["rm", "-rf", f"{hplt}/mt_en"])
    subprocess.call(["mkdir", "-p", f"{hplt}/mt_en"])
    mt_en = f"{hplt}/mt_en/mt_en"

    subprocess.call(["opus_read", "-s", "mt", "-t", "en", "-d", "HPLT", "-p", "moses", "--write",
     f"{mt_en}-mt.txt", f"{mt_en}-en.txt", "-q"])

    # delete zip file
    subprocess.call(["rm", "HPLT_latest_moses_en-mt.txt.zip"])

################################## mt / aeb <--> en ##################################
def ary_en_doda():
    subprocess.call(["rm", "-rf", f"{doda}/ary_en"])
    subprocess.call(["mkdir", "-p", f"{doda}/ary_en"])
    ary_en = f"{doda}/ary_en/ary_en"

    url = "https://raw.githubusercontent.com/darija-open-dataset/dataset/main/sentences/sentences.csv"
    response = requests.get(url)
    response.raise_for_status()

    with open("ary_en.csv", "wb") as f:
        f.write(response.content)

    df = pd.read_csv("ary_en.csv")

    df["darija"].to_csv(f"{ary_en}-ary_Latn.txt", index=False, header=False)
    df["eng"].to_csv(f"{ary_en}-en.txt", index=False, header=False)
    df["darija_ar"].to_csv(f"{ary_en}-ary_Arab.txt", index=False, header=False)

    subprocess.call(["rm", "ary_en.csv"])


    

################################## flores+ ##################################
def flores_plus():
    huggingface_hub.login(token=HF_TOKEN)

    subprocess.call(["rm", "-rf", flores])
    dev = f"{flores}/dev"
    devtest = f"{flores}/devtest"
    
    subprocess.call(["mkdir", "-p", dev])
    subprocess.call(["mkdir", "-p", devtest])
    
    languages = ["fra_Latn", "spa_Latn", "por_Latn", "eng_Latn", "arg_Latn", 
                "mfe_Latn", "uzn_Latn", "kaa_Latn", "ces_Latn", "deu_Latn", "amh_Ethi",
                "tir_Ethi", "fil_Latn", "ben_Beng", "mlt_Latn", "aeb_Arab", "ary_Arab",
                "cat_Latn", "oci_Latn", "mlt_Latn", "aeb_Arab", "ary_Arab"]

    for lang in languages:
        dataset = load_dataset("openlanguagedata/flores_plus", lang)

        try:
            dev_df = pd.DataFrame(dataset["dev"])
            dev_df["text"].to_csv(f"{dev}/{lang}.txt", index=False, header=False)
        except KeyError:
            print(f"no dev for {lang}")
        
        devtest_df = pd.DataFrame(dataset["devtest"])
        devtest_df["text"].to_csv(f"{devtest}/{lang}.txt", index=False, header=False)


################################## helper functions ##################################
def download_tmx_subset(url, output_path, max_bytes):
    wget_proc = subprocess.Popen(["wget", "-q", "-O", "-", url], stdout=subprocess.PIPE)
    gunzip_proc = subprocess.Popen(["gunzip", "-c"], stdin=wget_proc.stdout, stdout=subprocess.PIPE)

    wget_proc.stdout.close()

    bytes_written = 0
    chunk_size = 1024 * 1024

    pbar = tqdm(total=max_bytes, unit="B", unit_scale=True, desc="Downloading")

    with open(output_path, "wb") as f:
        try:
            while bytes_written < max_bytes:
                remaining = max_bytes - bytes_written
                to_read = min(chunk_size, remaining)

                chunk = gunzip_proc.stdout.read(to_read)
                if not chunk:
                    break

                f.write(chunk)
                bytes_written += len(chunk)
                pbar.update(len(chunk))
        
        finally:
            pbar.close()
            gunzip_proc.terminate()
            wget_proc.terminate()
        
    print(f"Downloaded {bytes_written} bytes to {output_path}")

def trunc_tmx(file_path):
    print("Truncating...")
    target = b"</tu>"
    chunk_size = 4096

    with open(file_path, "rb+") as f:
        f.seek(0, os.SEEK_END)
        file_size = f.tell()
        pointer = file_size

        while pointer > 0:
            step = min(pointer, chunk_size)
            pointer -= step
            f.seek(pointer)

            chunk = f.read(step)
            last_index = chunk.rfind(target)

            # found a </tu> tag to truncate at
            if last_index != -1:
                # truncate
                cut_point = pointer + last_index + len(target)
                f.seek(cut_point)
                f.truncate()

                # add footer tags
                f.write(b"\n  </body>\n</tmx>\n")
                return
    
    print(f"Error truncating {file_path}")

def extract_tmx_to_moses(tmx_path, src_lang, tgt_lang, src_output, tgt_output):
    with open(src_output, "w", encoding="utf-8") as f_src, \
         open(tgt_output, "w", encoding="utf-8") as f_tgt:
        
        context = etree.iterparse(tmx_path, events=("end",), tag="tu")
        
        count = 0
        with tqdm(context, desc="Extracting TMX", unit="seg") as pbar:
            for event, elem in pbar:
                segments = {src_lang: None, tgt_lang: None}
                
                for tuv in elem.findall("tuv"):
                    lang = tuv.get("{http://www.w3.org/XML/1998/namespace}lang")
                    seg = tuv.find("seg")
                    
                    if lang in segments and seg is not None:
                        segments[lang] = seg.text.strip() if seg.text else ""

                # only write if both sides of the translation exist
                if segments[src_lang] is not None and segments[tgt_lang] is not None:
                    f_src.write(segments[src_lang] + "\n")
                    f_tgt.write(segments[tgt_lang] + "\n")
                    count += 1

                # clear element and previous nodes from tree
                elem.clear()
                while elem.getprevious() is not None:
                    del elem.getparent()[0]

    print(f"{tmx_path} extraction complete. Total lines: {count}")



if __name__ == "__main__":

    # print("########### Download Uzbek -> English ###########")
    # process_nllb("eng_Latn", "uzn_Latn", "uz", "en", k=11000000, switch=True)
    # print("########### Download Karakalpak <--> English/Uzbek ###########")
    # kaa_uz_en_oldi()

    # print("########### Download Amharic / Tigrinya <--> English ###########")
    # process_nllb("amh_Ethi", "eng_Latn", "am", "en", k=8000000)
    # process_nllb("amh_Ethi", "tir_Ethi", "am", "ti", k=300472)
    # process_nllb("eng_Latn", "tir_Ethi", "ti", "en", k=700000, switch=True)

    # print("########### Download Tagalog <--> English ###########")
    # # NLLB
    # process_nllb("eng_Latn", "tgl_Latn", "tl", "en", k=11000000, switch=True)
    # # CCAligned + CCMatrix
    # process_ccmatrix_subset(1000, "tl", "en", "https://object.pouta.csc.fi/OPUS-CCMatrix/v1/tmx/en-tl.tmx.gz", trunc=False)
    # tl_en_ccalign()


    # print("########### Download Bikol <--> English ###########")
    # bik_en_wikimedia()

    # print("########### Download Spanish / Portuguese <--> English ###########")
    # process_ccmatrix_subset(3400, "es", "en", "https://object.pouta.csc.fi/OPUS-CCMatrix/v1/tmx/en-es.tmx.gz")
    # process_ccmatrix_subset(3400, "pt", "en", "https://object.pouta.csc.fi/OPUS-CCMatrix/v1/tmx/en-pt.tmx.gz")
    # process_ccmatrix_subset(3400, "es", "pt", "https://object.pouta.csc.fi/OPUS-CCMatrix/v1/tmx/es-pt.tmx.gz")

    # print("########### Download Aragonese <--> English/Spanish ###########")
    # es_an_en_wikimatrix()

    # print("########### Download French <--> English ###########")
    # process_ccmatrix_subset(3600, "fr", "en", "https://object.pouta.csc.fi/OPUS-CCMatrix/v1/tmx/en-fr.tmx.gz")
    # print("########### Download Mauritian Creole <--> English / French ###########")
    # mfe_en_fr_kreyolmt()
    # mfe_en_fr_kreolmorisienmt()

    # print("########### Download Seychellois Creole <--> English ###########")
    # crs_en_mt560()

    # print("########### Download Czech <--> German ###########")
    # process_ccmatrix_subset(3400, "cs", "de", "https://object.pouta.csc.fi/OPUS-CCMatrix/v1/tmx/cs-de.tmx.gz")
    # print("########### Download Upper Sorbian <--> German ###########")
    # hsb_de_wmt20()

    # print("########### Download Bengali <--> English ###########")
    # process_ccmatrix_subset(3500, "bn", "en", "https://object.pouta.csc.fi/OPUS-CCMatrix/v1/tmx/bn-en.tmx.gz", trunc=False)
    # print("########### Process Rohingya <--> English ###########")
    # rh_en_twb()

    # print("########### Download Catalan <--> English ###########")
    # process_nllb("cat_Latn", "eng_Latn", "ca", "en", k=11000000)
    # print("########### Download Occitan <--> English ###########")
    # process_nllb("eng_Latn", "oci_Latn", "oc", "en", switch=True, k=1730828)

    # print("########### Download Maltese <--> English ###########")
    # process_nllb("eng_Latn", "mlt_Latn", "mt", "en", switch=True, k=11000000)
    # mt_en_dgt()
    # mt_en_hplt()

    print("########### Download Maltese <--> English ###########")
    ary_en_doda()

    # print("########### Download Chavacano <--> English ###########")
    # cbk_en_chavmt() # waiting for access


    # print("########### Download FLORES+ ###########")
    # flores_plus()

