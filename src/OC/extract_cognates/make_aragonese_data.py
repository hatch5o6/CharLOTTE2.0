import csv
import random

random.seed(42)

# From Aragonese-English Parallel
an_en_train = "/home/hatch5o6/nobackup/archive/data/CharLOTTE_data/an-en/train.an.txt"
an_en_val = "/home/hatch5o6/nobackup/archive/data/CharLOTTE_data/an-en/val.an.txt"
an_en_test = "/home/hatch5o6/nobackup/archive/data/CharLOTTE_data/an-en/test.an.txt"

# From PILAR
pilar_crawled = "/home/hatch5o6/nobackup/archive/data/MONOLINGUAL/Aragonese/Sources/PILAR/aragonese/crawled.txt"
pilar_literary = "/home/hatch5o6/nobackup/archive/data/MONOLINGUAL/Aragonese/Sources/PILAR/aragonese/literary.txt"

# from Leipzig
leipzig = "/home/hatch5o6/nobackup/archive/data/MONOLINGUAL/Aragonese/Sources/Leipzig/arg_wikipedia_2021_30K/arg_wikipedia_2021_30K-sentences.txt"

def read_lines(f):
    print("reading", f)
    with open(f) as inf:
        lines = [l.strip() for l in inf.readlines()]
    return lines

def read_leipzig(f):
    print("LEIPZIG reading", f)
    # sents = []
    # with open(f) as inf:
    #     reader = csv.reader(inf, delimiter='\t')
    #     for id, sent in reader:
    #         sents.append(sent.strip())  
    sents = []
    lines = read_lines(f)
    for line in lines:
        _, _, s = line.partition("\t")
        sents.append(s)
    return sents

DO_NOT_INCLUDE = set(read_lines(an_en_val) + read_lines(an_en_test))
data_files = {
    an_en_train: set(read_lines(an_en_train)).difference(DO_NOT_INCLUDE),
    pilar_crawled: set(read_lines(pilar_crawled)).difference(DO_NOT_INCLUDE),
    pilar_literary: set(read_lines(pilar_literary)).difference(DO_NOT_INCLUDE),
    leipzig: set(read_leipzig(leipzig)).difference(DO_NOT_INCLUDE)
}

out = "/home/hatch5o6/nobackup/archive/data/MONOLINGUAL/Aragonese/Comprised/an.txt"
notes = out[:-3] + "notes"

with open(out, "w") as outf, open(notes, "w") as notf:
    notf.write(f"REMOVED ANY OVERLAP WITH:\n\t-`{an_en_val}`\n\t-`{an_en_test}`\n")
    notf.write("DATA FROM:\n")
    all_lines = set()
    total = 0
    for data_f, lines in data_files.items():
        total += len(lines)
        notf.write(f"\t-({len(lines)}) `{data_f}`\n")
        all_lines.update(lines)
    all_lines = sorted(all_lines)
    random.shuffle(all_lines)
    notf.write(f"UNIQUIFIED {total} >> {len(all_lines)}")
    outf.write("\n".join(all_lines) + "\n")
