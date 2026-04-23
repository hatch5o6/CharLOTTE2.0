from sacrebleu.metrics import BLEU, CHRF
import functools

def validate_lens(f):
    @functools.wraps(f)
    def wrapper(*args):
        hyp, ref = args[:2]
        if len(hyp) != len(ref):
            raise ValueError(f"Len hyp ({len(hyp)}) != len ref ({len(ref)})")
        score = f(*args)
        return score
    return wrapper

def validate_src_hyp_ref_lens(f):
    @functools.wraps(f)
    def wrapper(*args):
        src, hyp, ref = args[:3]
        if not (len(src) == len(hyp) == len(ref)):
            raise ValueError(f"Len src {len(src)} != len hyp ({len(hyp)}) != len ref ({len(ref)})")
        score = f(src, hyp, ref)
        return score
    return wrapper

def print_score(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        print(f"------------ {f.__name__} ------------")
        assert len(args) == 2
        print("args has hyp and ref")
        print("kwargs:")
        for k, v in kwargs.items():
            print(f"\t{k}=`{v}`")
        score = f(*args, **kwargs)
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
def calc_spBLEU(
    hyp,
    ref,
    tokenizer="flores200"
):
    print(f"spBLEU, tokenizer={tokenizer}")
    bleu = BLEU(tokenize=tokenizer)
    return bleu.corpus_score(hyp, [ref])

@print_score
@validate_lens
def calc_chrF(
    hyp,
    ref
):
    chrf = CHRF()
    return chrf.corpus_score(hyp, [ref])

@print_score
@validate_lens
def calc_chrF_plus_plus(
    hyp,
    ref
):
    chrf_plus_plus = CHRF(word_order=2)
    return chrf_plus_plus.corpus_score(hyp, [ref])

@validate_src_hyp_ref_lens
def calc_comet(
    src,
    hyp,
    ref
):
    pass

if __name__ == "__main__":
    hyp = ["Hello my frend", "How are you doing today?"]
    ref = ["Hello my friend", "How are you feeling this morning?"]
    chrf = calc_chrF(hyp, ref)
    bleu = calc_BLEU(hyp, ref)
    charbleu = calc_charBLEU(hyp, ref)
    