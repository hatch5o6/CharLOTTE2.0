from sloth_hatch.sloth import log_function_call

# spacy tokenizers
def _get_spacy_tokenizer(model='xx_sent_ud_sm'):
    import spacy
    nlp = spacy.load(model, exclude=["tagger", "parser", "ner", "lemmatizer", "textcat", "custom", "entity_linker", "entity_ruler", "textcat_multilabel", "trainable_lemmatizer", "morphologizer", "attribute_ruler", "senter", "sentencizer", "tok2vec", "transformer"])
    def spacy_tokenize(line):
        doc = nlp(line)
        return [tok.text for tok in doc]
    return spacy_tokenize

def multi_word_tokenizer():
    return _get_spacy_tokenizer('xx_sent_ud_sm')

def es_word_tokenizer():
    return _get_spacy_tokenizer('es_core_news_sm')


# nltk tokenizers
def _get_nltk_tokenizer(lang):
    import nltk
    from nltk.tokenize import word_tokenize

    try:
        nltk.data.find('tokenizers/punkt')
        print("punkt is already installed.")
    except:
        print("punkt not found. Downloading...")
        nltk.download('punkt')
    
    def nltk_tokenize(line):
        return word_tokenize(line.strip(), language=lang)

    return nltk_tokenize
    
def en_word_tokenizer():
    return _get_nltk_tokenizer("english")

def fr_word_tokenizer():
    return _get_nltk_tokenizer("french")


# indic tokenizers
def _get_indic_word_tokenizer():
    from indicnlp.tokenize import indic_tokenize
    def indic_word_tokenize(line):
        return indic_tokenize.trivial_tokenize_indic(line)
    return indic_word_tokenize

def indic_word_tokenizer():
    return _get_indic_word_tokenizer()


# camel tokenizers
def _get_ar_word_tokenizer(split_digits):
    from camel_tools.tokenizers.word import simple_word_tokenize as camel_simple_word_tokenize
    def ar_tokenize(line):
        return camel_simple_word_tokenize(line, split_digits=split_digits)
    return ar_tokenize

def ar_word_tokenizer():
    return _get_ar_word_tokenizer(split_digits=True)


# by lang
def get_tokenizer(lang):
    if lang in ["en", "djk"]:
        return en_word_tokenizer()
    elif lang in ["es", "an", "ast"]:
        return es_word_tokenizer()
    elif lang in ["fr", "oc", "mfe"]:
        return fr_word_tokenizer()
    elif lang in ["lua", "bem", "ewe", "fon", "multi"]:
        return multi_word_tokenizer()
    elif lang in ["ar", "aeb", "apc"]:
        return ar_word_tokenizer()
    elif lang in ["hi", "as", "bn", "bho"]:
        return indic_word_tokenizer()
    else:
        raise ValueError(f"language {lang} is not an option for tokenization!")

if __name__ == "__main__":
    try:
        non_tokenize = get_tokenizer("?")
    except ValueError as e:
        print("NON-TOKENIZE:", e)

    print("\nEN TOKENIZER")
    en_tokenize = get_tokenizer("en")
    print(en_tokenize("Hello, how are you?"))

    print("\nES TOKENIZER")
    es_tokenize = get_tokenizer("es")
    print(es_tokenize("¿Que estás haciendo con el «vaso» hoy?"))
    print(es_tokenize("-¿Que estás haciendo con el «vaso» hoy?"))
    print(es_tokenize("¡Que estás haciendo con el «vaso» hoy!"))
    print(es_tokenize("-¡Que estás haciendo con el «vaso» hoy!"))

    print("\nFR TOKENIZER")
    fr_tokenize = get_tokenizer("fr")
    print(fr_tokenize("Que fais-tu avec le «verre» aujourd'hui?"))

    print("\nAR TOKENIZER")
    ar_tokenize = get_tokenizer("ar")
    print(ar_tokenize("مرحبا، يا أصدقائي! ماذا تفعلون اليوم؟"))

    print("\nMULTI TOKENIZER")
    multi_tokenize = get_tokenizer("multi")
    print(multi_tokenize("¿Que estás haciendo con el «vaso» hoy?"))

