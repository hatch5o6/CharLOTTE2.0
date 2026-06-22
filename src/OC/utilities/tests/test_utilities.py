import pytest
import os
import shutil
from sloth_hatch import sloth

from OC.utilities import utilities

#######
# NLD #
#######
def test_NLD_zero():
    assert utilities.NLD("hello", "hello") == 0.0

def test_NLD_one():
    assert utilities.NLD("abcd", "efgh") == 1.0

def test_NLD():
    assert utilities.NLD("aaaa", "abaa") == 0.25
    assert utilities.NLD("aaaa", "abba") == 0.5
    assert utilities.NLD("aaaa", "abbab") == 0.6

#################
# write_oc_data #
#################

def test_write_oc_data():
    out_dir = "src/OC/utilities/tests/fixtures/sample_oc_out"
    sloth.create_directory(out_dir, destroy=True)
    assert os.listdir(out_dir) == []

    dataset = [
        (0.5, "zippidy doo daw", 4, "hungry"),
        (3.14, "huh???", 2341, "hippos", "meat"),
        ("theres", "a new", 2)
    ]

    out_path = os.path.join(out_dir, "test_out.txt")
    assert not os.path.exists(out_path)

    utilities.write_oc_data(dataset, out_path)

    assert sloth.read_lines(out_path) == [
        "0.5 ||| zippidy doo daw ||| 4 ||| hungry",
        "3.14 ||| huh??? ||| 2341 ||| hippos ||| meat",
        "theres ||| a new ||| 2"
    ]
    shutil.rmtree(out_dir)


################
# read_oc_data #
################
    
def test_read_parallel_oc_data():
    f = "src/OC/utilities/tests/fixtures/sample_oc_data/parallel_cogs.txt"
    assert utilities.read_oc_data(f) == [
        (4, "legión", "lechion", 0.43),
        (2, "casa", "masa", 0.25),
        (7, "hombre", "hommen", 0.5)
    ]

def test_read_fuzzy_oc_data():
    f = "src/OC/utilities/tests/fixtures/sample_oc_data/fuzzy_cogs.txt"
    assert utilities.read_oc_data(f) == [
        (4, 1, "legión", "lechion", 0.43),
        (2, 3, "casa", "masa", 0.25),
        (7, 3, "hombre", "hommen", 0.5)
    ]

def test_read_mixed_oc_data():
    f = "src/OC/utilities/tests/fixtures/sample_oc_data/mixed_cogs.txt"
    with pytest.raises(AssertionError, match=r"Inconsistent tuple sizes! Should be 5 but got 4!"):
        utilities.read_oc_data(f)

def test_read_bad_oc_data():
    f = "src/OC/utilities/tests/fixtures/sample_oc_data/bad_cogs.txt"
    with pytest.raises(AssertionError, match=r"OC tuples must be of size 4 \(freq, w1, w2, dist\) or 5 \(freq1, freq2, w1, w2, dist\)\. Got 6!"):
        utilities.read_oc_data(f)