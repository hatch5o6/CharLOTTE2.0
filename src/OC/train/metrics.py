from sacrebleu.metrics import BLEU, CHRF
import functools

def validate_lens(f):
    @functools.wraps(f)
    def wrapper(hyp, ref):
        if len(hyp) != len(ref):
            raise ValueError(f"Len hyp ({len(hyp)}) != len ref({len(ref)})")
        score = f(hyp, ref)
        return score
    return wrapper

def print_score(f):
    @functools.wraps(f)
    def wrapper(hyp, ref):
        print(f"------------ {f.__name__} ------------")
        score = f(hyp, ref)
        print(score)
        print(score.score)
        return score
    return wrapper

@print_score
@validate_lens
def calc_charBLEU(
    hyp,
    ref
):
    bleu = BLEU(tokenize="char")
    return bleu.corpus_score(hyp, [ref])

@print_score
@validate_lens
def calc_BLEU(
    hyp,
    ref
):
    bleu = BLEU()
    return bleu.corpus_score(hyp, [ref])

@print_score
@validate_lens
def calc_chrF(
    hyp,
    ref
):
    chrf = CHRF()
    return chrf.corpus_score(hyp, [ref])


if __name__ == "__main__":
    hyp = ["Hello my frend", "How are you doing today?"]
    ref = ["Hello my friend", "How are you feeling this morning?"]
    chrf = calc_chrF(hyp, ref)
    bleu = calc_BLEU(hyp, ref)
    charbleu = calc_charBLEU(hyp, ref)
    