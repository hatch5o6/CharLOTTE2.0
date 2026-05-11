import sys
from rapidfuzz import process, fuzz
from aksharamukha import transliterate as transliterate_ak
from indic_transliteration import sanscript
import uroman
from scipy.spatial.distance import jensenshannon
import numpy as np
from collections import Counter

# filepaths
DATA_HOME = sys.argv[1]

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


def main():
    rhg_lines = readfile(f"{twb}/rhg_en/cleaned/src.txt")
    rhg_bag = make_bag(rhg_lines)
    bn_lines = readfile(f"{ccmat}/bn_en/bn_en-bn.txt")
    ak_romanizers = ["ITRANS", "Velthuis", "IAST", "ISO", "Titus", 
                    "SLP1", "WX", "RomanReadable", "RomanColloquial"]
    it_romanizers = [sanscript.IAST, sanscript.ITRANS, sanscript.HK]


    results = {}
    
    # aksharamukha
    for romanizer in ak_romanizers:
        bn_Latn = romanize_ak(bn_lines, romanizer)
        bn_bag = make_bag(bn_Latn)
        print(f"ak_{romanizer}")
        print(bn_bag[10:20])
        results[f"ak_{romanizer}"] = jensen_shannon_divergence(rhg_bag, bn_bag)
    
    # indic-transliterate
    for romanizer in it_romanizers:
        bn_Latn = romanize_it(bn_lines, romanizer)
        bn_bag = make_bag(bn_Latn)
        print(f"it_{str(romanizer)}")
        print(bn_bag[10:20])
        results[f"it_{str(romanizer)}"] = jensen_shannon_divergence(rhg_bag, bn_bag)

    # uroman
    bn_Latn = romanize_ur(bn_lines)
    bn_bag = make_bag(bn_Latn)
    print(f"ur_standard")
    print(bn_bag[10:20])
    results[f"ur_standard"] = jensen_shannon_divergence(rhg_bag, bn_bag)

    for key in results:
        print(key, round(results[key], 3))
    
    best = min(results, key=results.get)
    print(f"BEST: {best, round(results[best], 3)}")



def jensen_shannon_divergence(bag1, bag2):
    # 1. Count frequencies
    counts1 = Counter(bag1)
    counts2 = Counter(bag2)
    
    # 2. Create a unified vocabulary
    vocal = sorted(list(set(counts1.keys()) | set(counts2.keys())))
    
    # 3. Convert to frequency vectors
    # We must ensure both vectors are the same length and align by word index
    vec1 = np.array([counts1.get(word, 0) for word in vocal])
    vec2 = np.array([counts2.get(word, 0) for word in vocal])
    
    # 4. Normalize to create probability distributions (sum to 1)
    p = vec1 / vec1.sum()
    q = vec2 / vec2.sum()
    
    # 5. Calculate JSD (scipy returns the square root / JS Distance)
    # To get the actual Divergence, square the result.
    js_distance = jensenshannon(p, q)
    js_divergence = js_distance**2
    
    return js_divergence

def romanize_ak(lines, tgt):
    Latn = []
    for line in lines:
        Latn.append(transliterate_ak.process("Bengali", tgt, line))
    return Latn

def romanize_it(lines, tgt):
    Latn = []
    for line in lines:
        Latn.append(sanscript.transliterate(line, sanscript.BENGALI, tgt))
    return Latn

def romanize_ur(lines):
    ur = uroman.Uroman()
    Latn = []
    for line in lines:
        Latn.append(ur.romanize_string(line, lcode='ben'))
    return Latn

def matches():
    rhg_tgt = readfile(f"{twb}/rhg_en/cleaned/tgt.txt")
    print(rhg_tgt[:5])

    bn_tgt = readfile(f"{ccmat}/bn_en/bn_en-en.txt")
    print(bn_tgt[:5])

    matches = set(bn_tgt).intersection(rhg_tgt)
    print(f"Matches: {len(matches)}")
    print(matches)

def readfile(filename):
    with open(filename, "r") as file:
        lines = [line.strip() for line in file.readlines()]
    return lines

def make_bag(lines):
    bag = []
    for line in lines:
        for word in line.split(' '):
            bag.append(word)
    return bag


if __name__ == "__main__":
    main()