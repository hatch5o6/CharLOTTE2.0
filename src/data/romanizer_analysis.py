from utilities.utilities import set_env
from rapidfuzz import process, fuzz
from aksharamukha import transliterate as transliterate_ak
from indic_transliteration import sanscript
import uroman
from camel_tools.utils.dediac import dediac_ar
from camel_tools.utils.charmap import CharMapper
from camel_tools.utils.transliterate import Transliterator
from lang_trans.arabic import buckwalter, arabtex, iso233
import romanize3
from scipy.spatial.distance import jensenshannon
import numpy as np
from collections import Counter
import os

# filepaths
set_env()
DATA_HOME = os.environ["DATA_HOME"]

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


def bengali_analysis():
    rhg_lines = readfile(f"{twb}/rhg_en/cleaned/src.txt")
    rhg_bag = make_bag_types(rhg_lines)
    bn_lines = readfile(f"{ccmat}/bn_en/cleaned/src.txt")
    ak_romanizers = ["ITRANS", "Velthuis", "IAST", "ISO", "Titus", 
                    "SLP1", "WX", "RomanReadable", "RomanColloquial"]
    it_romanizers = [sanscript.IAST, sanscript.ITRANS, sanscript.HK]

    print("data loaded")
    results = {}
    
    print("Bengali Example Sentence:")
    print(bn_lines[6])
    # aksharamukha
    for romanizer in ak_romanizers:
        bn_Latn = romanize_ak(bn_lines, romanizer)
        bn_bag = make_bag_types(bn_Latn)
        print(f"ak_{romanizer}")
        print(bn_Latn[6])
        results[f"ak_{romanizer}"] = jensen_shannon_divergence(rhg_bag, bn_bag)
    
    # indic-transliterate
    for romanizer in it_romanizers:
        bn_Latn = romanize_it(bn_lines, romanizer)
        bn_bag = make_bag_types(bn_Latn)
        print(f"it_{str(romanizer)}")
        print(bn_Latn[6])
        results[f"it_{str(romanizer)}"] = jensen_shannon_divergence(rhg_bag, bn_bag)

    # uroman
    bn_Latn = romanize_ur(bn_lines, 'ben')
    bn_bag = make_bag_types(bn_Latn)
    print(f"ur_standard")
    print(bn_Latn[6])
    results[f"ur_standard"] = jensen_shannon_divergence(rhg_bag, bn_bag)


    print("\n\nResults Summary")
    for key in results:
        print(key, round(results[key], 3))
    
    best = min(results, key=results.get)
    print(f"BEST: {best, round(results[best], 3)}")

def tunisian_analysis():
    mt_lines = readfile(f"{nllb}/mt_en/cleaned/src.txt")
    mt_lines = remove_vowels(mt_lines)
    mt_bag = make_bag_types(mt_lines)

    aeb_lines = readfile(f"{ldc}/aeb_en/cleaned/src.txt")
    aeb_lines = dediac_lines(aeb_lines)
    ct_romanizers = ["ar2bw", "ar2safebw", "ar2xmlbw", "ar2hsb"]
    lt_romanizers = ["buckwalter", "arabtex", "iso233"]
    print("data loaded")
    results = {}
    
    print("Tunisian Example Sentence:")
    print(aeb_lines[2])


    # camel-tools
    for romanizer in ct_romanizers:
        aeb_Latn = romanize_ct(aeb_lines, romanizer)
        aeb_bag = make_bag_types(aeb_Latn)
        print(f"ct_{romanizer}")
        print(aeb_Latn[2])
        results[f"ct_{romanizer}"] = jensen_shannon_divergence(mt_bag, aeb_bag)

    # lang-trans
    for romanizer in lt_romanizers:
        aeb_Latn = romanize_lt(aeb_lines, romanizer)
        aeb_bag = make_bag_types(aeb_Latn)
        print(f"lt_{romanizer}")
        print(aeb_Latn[2])
        results[f"lt_{romanizer}"] = jensen_shannon_divergence(mt_bag, aeb_bag)
    
    # uroman - slow by comparison to the others
    aeb_Latn = romanize_ur(aeb_lines, 'ara')
    aeb_bag = make_bag_types(aeb_Latn)
    print(f"ur_standard")
    print(aeb_Latn[2])
    results[f"ur_standard"] = jensen_shannon_divergence(mt_bag, aeb_bag)

    # romanize3
    aeb_Latn = romanize_r3(aeb_lines)
    aeb_bag = make_bag_types(aeb_Latn)
    print(f"romanize3")
    print(aeb_Latn[2])
    results[f"romanize3"] = jensen_shannon_divergence(mt_bag, aeb_bag)

    print("\n\nResults Summary")
    for key in results:
        print(key, results[key])

    best = min(results, key=results.get)
    print(f"BEST: {best, round(results[best], 3)}")


def moroccan_analysis():
    mt_lines = readfile(f"{nllb}/mt_en/cleaned/src.txt")
    mt_lines = remove_vowels(mt_lines)
    mt_bag = make_bag(mt_lines)

    ary_lines = readfile(f"{doda}/ary_en/ary_en-ary_Arab.txt")
    ary_lines = dediac_lines(ary_lines)
    ct_romanizers = ["ar2bw", "ar2safebw", "ar2xmlbw", "ar2hsb"]
    lt_romanizers = ["buckwalter", "arabtex", "iso233"]
    print("data loaded")
    results = {}
    print("Moroccan Example Sentence:")
    print(ary_lines[14])

    # camel-tools
    for romanizer in ct_romanizers:
        ary_Latn = romanize_ct(ary_lines, romanizer)
        ary_bag = make_bag(ary_Latn)
        print(f"ct_{romanizer}")
        print(ary_Latn[14])
        results[f"ct_{romanizer}"] = jensen_shannon_divergence(mt_bag, ary_bag)

    # lang-trans
    for romanizer in lt_romanizers:
        ary_Latn = romanize_lt(ary_lines, romanizer)
        ary_bag = make_bag(ary_Latn)
        print(f"lt_{romanizer}")
        print(ary_Latn[14])
        results[f"lt_{romanizer}"] = jensen_shannon_divergence(mt_bag, ary_bag)
    
    # uroman - slow by comparison to the others
    ary_Latn = romanize_ur(ary_lines, 'ara')
    ary_bag = make_bag(ary_Latn)
    print(f"ur_standard")
    print(ary_Latn[14])
    results[f"ur_standard"] = jensen_shannon_divergence(mt_bag, ary_bag)

    # romanize3
    ary_Latn = romanize_r3(ary_lines)
    ary_bag = make_bag(ary_Latn)
    print(f"romanize3")
    print(ary_Latn[14])
    results[f"romanize3"] = jensen_shannon_divergence(mt_bag, ary_bag)

    # DODa
    ary_Latn = readfile(f"{doda}/ary_en/ary_en-ary_Latn.txt")
    ary_bag = make_bag(ary_Latn)
    print(f"DODa")
    print(ary_Latn[14])
    results[f"DODa"] = jensen_shannon_divergence(mt_bag, ary_bag)

    print("\n\nResults Summary")
    for key in results:
        print(key, results[key])

    best = min(results, key=results.get)
    print(f"BEST: {best, round(results[best], 3)}")
    

def jensen_shannon_divergence(bag1, bag2):
    # Count frequencies
    counts1 = Counter(bag1)
    counts2 = Counter(bag2)
    
    # Create a unified vocabulary
    vocal = sorted(list(set(counts1.keys()) | set(counts2.keys())))
    
    # Convert to frequency vectors
    # We must ensure both vectors are the same length and align by word index
    vec1 = np.array([counts1.get(word, 0) for word in vocal])
    vec2 = np.array([counts2.get(word, 0) for word in vocal])
    
    # Normalize to create probability distributions (sum to 1)
    p = vec1 / vec1.sum()
    q = vec2 / vec2.sum()
    
    # Calculate JSD (scipy returns the square root / JS Distance)
    js_distance = jensenshannon(p, q)
    js_divergence = js_distance**2
    
    return js_divergence


### Helper Functions ###
def matches():
    mt_tgt = readfile(f"{nllb}/mt_en/cleaned/tgt.txt")
    print(len(mt_tgt))
    print(mt_tgt[:5])

    ary_tgt = readfile(f"{doda}/ary_en/ary_en-en.txt")
    print(len(ary_tgt))
    print(ary_tgt[:5])

    matches = set(mt_tgt).intersection(ary_tgt)
    print(f"Matches: {len(matches)}")
    print(matches)

def readfile(filename):
    with open(filename, "r") as file:
        lines = [line.lower().strip() for line in file.readlines()]
    if len(lines) < 500000:
        return lines
    else:
        return lines[:500000]

def make_bag(lines):
    bag = []
    for line in lines:
        for word in line.split(' '):
            bag.append(word)
    return bag

def make_bag_types(lines):
    bag = set()
    for line in lines:
        for word in line.split(' '):
            bag.add(word)
    return list(bag)

def remove_vowels(lines):
    nlines = []
    for line in lines:
        nline = line.replace("a", "").replace("e", "").replace("i", "").replace("o", "").replace("u", "")
        nline = nline.replace("A", "").replace("E", "").replace("I", "").replace("O", "").replace("U", "")
        nlines.append(nline)
    return nlines

def dediac_lines(lines):
    nlines = []
    for line in lines:
        nline = dediac_ar(line)
        nlines.append(nline)
    return nlines
        

### Bengali Romanizers ###
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

def romanize_ur(lines, lang):
    ur = uroman.Uroman()
    Latn = []
    for line in lines:
        Latn.append(ur.romanize_string(line, lcode=lang))
    return Latn

### Tunisian Romanizers ###
def romanize_ct(lines, map):
    mapper = CharMapper.builtin_mapper(map)
    translit = Transliterator(mapper)
    Latn = []
    for line in lines:
        Latn.append(translit.transliterate(line))
    return Latn

def romanize_lt(lines, map):
    Latn = []
    d = {"buckwalter":buckwalter, "arabtex":arabtex, "iso233":iso233}
    for line in lines:
        Latn.append(d[map].transliterate(line))
    return Latn

def romanize_r3(lines):
    Latn = []
    r = romanize3.__dict__["ara"]
    for line in lines:
        Latn.append(r.convert(line))
    return Latn

if __name__ == "__main__":
    # bengali_analysis()
    # tunisian_analysis()
    moroccan_analysis()
    # matches()

    # aeb_line = ["أنا قاعد في ال- في السنتر فيل شارع غانة"]
    # print(aeb_line)
    # print(dediac_lines(aeb_line))
    # print(romanize_ur(aeb_line, 'ara'))
    
    # mt_line = ["Dawn jemmnu li Alla qed jgħin lill-Iżraelin, u ma jridux jiġġieldu kontra Alla."]
    # print(mt_line)
    # print(remove_vowels(mt_line))

