from OC.utilities.word_tokenizers import get_tokenizer

def test_ws_tokenizer():
    tokenizer = get_tokenizer("ws")
    sentence = "Hey there, $partner,\nwhat're you up    to\ttoday?"
    assert tokenizer(sentence) == ["Hey", "there,", "$partner,", "what're", "you", "up", "to", "today?"] == sentence.split()

