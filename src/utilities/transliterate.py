from utilities.utilities import set_env
import os
import subprocess
from tqdm import tqdm
import argparse
from aksharamukha import transliterate as transliterate_ak
import uroman


set_env()
DATA_HOME = os.environ["DATA_HOME"]


def transliterate_bengali(transliterator):
    bn_en_f = f"{DATA_HOME}/CharLOTTE_data/bn-en"

    for split in ['train', 'test', 'val']:
        subprocess.call(["mv", f"{bn_en_f}/{split}.bn.txt", f"{bn_en_f}/{split}.orig.bn.txt"])
        bn_lines = readfile(f"{bn_en_f}/{split}.orig.bn.txt")

        ak_romanizers = ["ITRANS", "Velthuis", "IAST", "ISO", "Titus", 
                    "SLP1", "WX", "RomanReadable", "RomanColloquial"]
        
        if transliterator in ak_romanizers:
            bn_Latn = romanize_ak(bn_lines, "RomanColloquial")

        with open(f"{bn_en_f}/{split}.bn.txt", "w") as outfile:
            for line in bn_Latn:
                outfile.write(line + '\n')
    

def transliterate_arabic(word_pairs, transliterator):
    word_pairs_Latn = []
    ur = uroman.Uroman()
    for pl_word, cl_word in word_pairs:
        if transliterator == 'uroman':
            word_pairs_Latn.append((pl_word, ur.romanize_string(cl_word, lcode='ara')))
    return word_pairs_Latn


def readfile(filename):
    with open(filename, "r") as file:
        lines = [line.lower().strip() for line in file.readlines()]
    return lines

def romanize_ak(lines, romanizer):
    Latn = []
    for line in tqdm(lines):
        Latn.append(transliterate_ak.process("Bengali", romanizer, line))
    return Latn



# def get_args():
#     parser = argparse.ArgumentParser(description="Transliterate Data")

#     parser.add_argument('--language', '-l', type=str, choices=['bn', 'aeb', 'ary'], help="'bn' is a parent language, 'aeb' and 'ary' are child languages")


#     TRANSLITERATORS = ["ITRANS", "Velthuis", "IAST", "ISO", "Titus", 
#                     "SLP1", "WX", "RomanReadable", "RomanColloquial",
#                     "uroman"]
    
#     parser.add_argument('--transliterator', '-t', type=str, choices=TRANSLITERATORS, help="use ['ITRANS', 'Velthuis', 'IAST', 'ISO', Titus', 'SLP1', 'WX', 'RomanReadable', 'RomanColloquial'] for 'bn', use ['uroman'] for 'aeb' and 'ary'")

#     return parser.parse_args()

# if __name__ == "__main__":
#     args = get_args()

#     if args.language == "bn":
#         transliterate_bengali(args.transliterator)
    
#     elif args.language == "ary" or "aeb":
#         transliterate_arabic(args.language, args.transliterator)