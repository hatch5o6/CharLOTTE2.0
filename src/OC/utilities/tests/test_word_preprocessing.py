import pytest

from OC.utilities import word_preprocessing as WP

#########
# clean #
#########

def test_clean():
    assert WP.clean("hello", long_enough=4) == "hello"
    assert WP.clean("$hello", long_enough=4) == "hello"
    assert WP.clean("$$hello", long_enough=4) == "hello"
    assert WP.clean("#hello", long_enough=4) == "hello"
    assert WP.clean("..hello", long_enough=4) == "hello"
    assert WP.clean("##hello", long_enough=4) == "hello"
    assert WP.clean(".hello", long_enough=4) == "hello"
    assert WP.clean("hello.", long_enough=4) == "hello"
    assert WP.clean("hello..", long_enough=4) == "hello"
    assert WP.clean("#hello..", long_enough=4) == "hello"
    assert WP.clean("#hello ..", long_enough=4) == "hello"
    assert WP.clean("# hello . .", long_enough=4) == "hello"
    assert WP.clean("#1111..", long_enough=4) == "1111"
    assert WP.clean("# he llo . .", long_enough=4) == "he llo"
    assert WP.clean("# he#llo . .", long_enough=4) == "he#llo"
    assert WP.clean("# he$llo . .", long_enough=4) == "he$llo"

def test_clean_len_zero():
    assert WP.clean("#", long_enough=4) == None
    assert WP.clean(".", long_enough=4) == None
    assert WP.clean("$", long_enough=4) == None
    assert WP.clean("#.$", long_enough=4) == None
    assert WP.clean(" ", long_enough=4) == None
    assert WP.clean("", long_enough=4) == None


def test_clean_too_short():
    assert WP.clean("hey", long_enough=4) == None
    assert WP.clean("$$$hey", long_enough=4) == None
    assert WP.clean("..hey.", long_enough=4) == None
    assert WP.clean(".$$hey..", long_enough=4) == None
    assert WP.clean(".$$hey..", long_enough=3) == "hey"
    assert WP.clean(".$$hey..", long_enough=2) == "hey"
    assert WP.clean(".$$hey..", long_enough=1) == "hey"
    assert WP.clean(".$$hey..", long_enough=0) == "hey"


###################
# _removable_char #
###################

def test_removable_char():
    assert WP._removable_char("a") == False
    assert WP._removable_char("5") == False
    assert WP._removable_char(".") == True
    assert WP._removable_char("?") == True
    assert WP._removable_char("$") == True
    assert WP._removable_char(" ") == True

    with pytest.raises(AssertionError, match="char must be a string!"):
        WP._removable_char(3)
    with pytest.raises(AssertionError, match="char must be a string of length 1!"):
        WP._removable_char("ab")

##################
# _is_only_punct #
##################

def test_is_only_punct():
    assert WP._is_only_punct("hello") == False
    assert WP._is_only_punct("hello?") == False
    assert WP._is_only_punct("hello$") == False
    assert WP._is_only_punct("hello?there") == False
    assert WP._is_only_punct("hello$there") == False
    assert WP._is_only_punct("$?") == True
    assert WP._is_only_punct("?$") == True
    assert WP._is_only_punct("$") == True
    assert WP._is_only_punct("?") == True

################
# _is_len_zero #
################

def test_is_len_zero():
    assert WP._is_len_zero("hey") == False
    assert WP._is_len_zero("he") == False
    assert WP._is_len_zero("h") == False
    assert WP._is_len_zero("") == True

##############
# _too_short #
##############

def test_too_short():
    assert WP._too_short("hello", threshold=0) == False
    assert WP._too_short("hello", threshold=3) == False
    assert WP._too_short("hello", threshold=4) == False
    assert WP._too_short("hello", threshold=5) == False
    assert WP._too_short("hello", threshold=6) == True
    assert WP._too_short("hello", threshold=7) == True

