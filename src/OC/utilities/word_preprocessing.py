from string import punctuation
punctuation += "—¡¿؟؛،٪»«›‹”“〞❮❯❛❟"

def clean(word):
    while len(word) > 0 and _removable_char(word[0]):
        word = word[1:]
    while len(word) > 0 and _removable_char(word[-1]):
        word = word[:-1]
    return word.strip()

def _removable_char(char):
    return char in punctuation or char.isspace()

def is_only_punct(word):
    global punctuation
    word = word.strip()
    for char in word:
        if char not in punctuation:
            return False
    return True
