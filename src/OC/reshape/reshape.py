from OC.utilities.word_tokenizers import get_tokenizer
from sloth_hatch.sloth import read_lines

from OC.utilities.word_preprocessing import clean

def prepare_source_words(pl_file, lang):
    word_tokenizer = get_tokenizer(lang)
    words = set()
    for line in read_lines(pl_file):
        words.update(word_tokenizer(line))
    words = [clean(w) for w in words]
    words = [w for w in words if (w != "") and (w != None)]
    words.sort()
    return words

def reshape_data(pl_file, word_mappings, lang, output_file):
    word_tokenizer = get_tokenizer(lang)
    reshaped_lines = []
    unique_orig_words = set()
    not_in_mappings = set()
    og_lines = read_lines(pl_file)
    for line in og_lines:
        tokens = line.split()
        for t, token in enumerate(tokens):
            words = word_tokenizer(token)
            for w, word in enumerate(words):
                cleaned_word = clean(word)
                unique_orig_words.add(cleaned_word)
                assert cleaned_word in word
                result = word_mappings.get(cleaned_word)
                if result:
                    word = word.replace(cleaned_word, result)
                else:
                    not_in_mappings.add(cleaned_word)
                words[w] = word
            tokens[t] = "".join(words)
        reshaped_lines.append(" ".join(tokens).strip())
    assert len(reshaped_lines) == len(og_lines)
    
    print(f"NOT IN WORD MAPPINGS ({len(not_in_mappings)})")
    for item in not_in_mappings:
        print(f"\t-`{item}`")

    print("UNIQUE WORDS IN ORIG DATA:", len(unique_orig_words))
    print("WORDS IN MAPPINGS:", len(word_mappings))

    with open(output_file, "w") as outf:
        outf.write("\n".join(reshaped_lines) + "\n")
