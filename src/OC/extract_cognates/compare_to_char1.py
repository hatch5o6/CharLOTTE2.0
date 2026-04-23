
def main():
    char1_cognates_f = "/home/hatch5o6/nobackup/archive/data/COGNATE_TRAIN_CoNLL/es-an_ES-AN-RNN-0_RNN-213_S-0/fastalign/word_list.es-an.NG.cognates.0.5.txt"
    char2_cognates_f = "/home/hatch5o6/nobackup/archive/CharLOTTE2.0_EXP/xx_xx-->xx_charlotte/OC/es_an/data/es_an.cognates.cognate_list"

    char1_cognates = parse_char1_cognate_list(char1_cognates_f)
    print("char1 before:", len(char1_cognates))
    char1_cognates = set(char1_cognates)
    print("char1 after:", len(char1_cognates), "\n\n")

    char2_cognates = parse_char2_cognate_list(char2_cognates_f)
    print("char2 before:", len(char2_cognates))
    char2_cognates = set(char2_cognates)
    print("char2 after:", len(char2_cognates), "\n\n")

    print("Are the same?")
    print(char1_cognates == char2_cognates, "\n\n")

    print("char1 - char2")
    print(len(char1_cognates.difference(char2_cognates)), "\n\n")

    print("char2 - char1")
    print(len(char2_cognates.difference(char1_cognates)), "\n\n")

    



def read_lines(f):
    with open(f) as inf:
        lines = [l.rstrip() for l in inf.readlines()]
    return lines

def parse_char2_cognate_list(f):
    print("reading", f)
    lines = read_lines(f)
    cognates = [eval(item) for item in lines]
    cognates.sort()
    return cognates

def parse_char1_cognate_list(f):
    print("reading", f)
    lines = read_lines(f)
    cognates = []
    for line in lines:
        freq, w1, w2, nld = line.split(" ||| ")
        freq = int(freq.strip())
        w1 = w1.strip()
        w2 = w2.strip()
        nld = float(nld.strip())
        cognates.append((freq, w1, w2, nld))
    cognates.sort()
    return cognates

if __name__ == "__main__":
    main()
