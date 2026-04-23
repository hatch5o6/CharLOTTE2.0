from string import punctuation
punctuation += "—¡¿؟؛،٪»«›‹”“〞❮❯❛❟"

def clean(word, long_enough):
    while len(word) > 0 and _removable_char(word[0]):
        word = word[1:]
    while len(word) > 0 and _removable_char(word[-1]):
        word = word[:-1]
    word = word.strip()
    if _is_len_zero(word):
        return None
    if _is_only_punct(word):
        return None
    if _too_short(word, threshold=long_enough):
        return None
    return word

def _removable_char(char):
    return char in punctuation or char.isspace()

def _is_only_punct(word):
    global punctuation
    word = word.strip()
    for char in word:
        if char not in punctuation:
            return False
    return True

def _is_len_zero(word):
    return len(word) == 0

def _too_short(word, threshold):
    return len(word) < threshold
