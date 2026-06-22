import pytest

from OC.train.CognateDataset import CognateDataset

def test_parallel_cognate_dataset_from_file():
    dataset = CognateDataset("src/OC/train/tests/fixtures/parallel_cognates.txt")
    assert len(dataset) == 9
    items = [item for item in dataset]
    assert items == [
        ("animals", "animales"),
        ("much", "mucho"),
        ("miracles", "milagros"),
        ("elephants", "elefantes"),
        ("1,000", "1,000"),
        ("555-444-0001", "555-444-0001"),
        ("555-434-0031", "555-434-0031"),
        ("2454", "2454"),
        ("3124", "3124")
    ]

def test_fuzzy_cognate_dataset_from_file():
    dataset = CognateDataset("src/OC/train/tests/fixtures/fuzzy_cognates.txt")
    assert len(dataset) == 9
    items = [item for item in dataset]
    assert items == [
        ("animals", "animales"),
        ("much", "mucho"),
        ("miracles", "milagros"),
        ("elephants", "elefantes"),
        ("1,000", "1,000,000"),
        ("555-444-0001", "555-444-0001"),
        ("555-434-0031", "555-434-0031"),
        ("2454", "2454"),
        ("3124", "3124")
    ]

def test_cognate_dataset_mixed_file():
    with pytest.raises(AssertionError, match="Inconsistent tuple sizes! Should be 5 but got 4!"):
        dataset = CognateDataset("src/OC/train/tests/fixtures/mixed_cognates.txt")
    

def test_cognate_dataset_bad_file():
    with pytest.raises(AssertionError, match=r"OC tuples must be of size 4 \(freq, w1, w2, dist\) or 5 \(freq1, freq2, w1, w2, dist\)\. Got 6!"):
        dataset = CognateDataset("src/OC/train/tests/fixtures/bad_cognates.txt")

def test_cognate_dataset_file_not_found():
    with pytest.raises(FileExistsError, match=r"Could not find file OC data path `src/OC/train/tests/fixtures/does_not_exist\.txt`"):
        dataset = CognateDataset("src/OC/train/tests/fixtures/does_not_exist.txt")

def test_cognate_dataset_from_data():
    data = [
        ("animals", "animales"),
        ("much", "mucho"),
        ("miracles", "milagros"),
        ("elephants", "elefantes"),
        ("1,000", "1,000,000"),
        ("555-444-0001", "555-444-0001"),
        ("555-434-0031", "555-434-0031"),
        ("2454", "2454"),
        ("3124", "3124")
    ]
    dataset = CognateDataset(data)
    assert len(dataset) == len(data) == 9
    items = [item for item in dataset]
    assert items == data

def test_cognate_dataset_invalid():
    data = (
        ("animals", "animales"),
        ("much", "mucho"),
        ("miracles", "milagros"),
        ("elephants", "elefantes"),
        ("1,000", "1,000,000"),
        ("555-444-0001", "555-444-0001"),
        ("555-434-0031", "555-434-0031"),
        ("2454", "2454"),
        ("3124", "3124")
    )
    with pytest.raises(ValueError, match=r"Data must be a list of \(src_word, tgt_word\) tuples!"):
        dataset = CognateDataset(data)


    data = [
        ["animals", "animales"],
        ("much", "mucho"),
        ("miracles", "milagros"),
        ("elephants", "elefantes"),
        ("1,000", "1,000,000"),
        ("555-444-0001", "555-444-0001"),
        ("555-434-0031", "555-434-0031"),
        ("2454", "2454"),
        ("3124", "3124")
    ]
    with pytest.raises(ValueError, match=r"Data must be a list of \(src_word, tgt_word\) tuples!"):
        dataset = CognateDataset(data)

    
    data = [
        ("animals", "animales", "danimals"),
        ("much", "mucho"),
        ("miracles", "milagros"),
        ("elephants", "elefantes"),
        ("1,000", "1,000,000"),
        ("555-444-0001", "555-444-0001"),
        ("555-434-0031", "555-434-0031"),
        ("2454", "2454"),
        ("3124", "3124")
    ]
    with pytest.raises(ValueError, match=r"Data must be a list of \(src_word, tgt_word\) tuples!"):
        dataset = CognateDataset(data)
    

    data = [
        ("animals", 6),
        ("much", "mucho"),
        ("miracles", "milagros"),
        ("elephants", "elefantes"),
        ("1,000", "1,000,000"),
        ("555-444-0001", "555-444-0001"),
        ("555-434-0031", "555-434-0031"),
        ("2454", "2454"),
        ("3124", "3124")
    ]
    with pytest.raises(ValueError, match=r"Data must be a list of \(src_word, tgt_word\) tuples!"):
        dataset = CognateDataset(data)
    

    data = [
        ("animals", "animales"),
        ("much", "mucho"),
        ("miracles", "milagros"),
        ("elephants", "elefantes"),
        ("1,000", "1,000,000"),
        ("555-444-0001", "555-444-0001"),
        ("555-434-0031", "555-434-0031"),
        ("2454", "2454"),
        ("3124", "3124")
    ]
    dataset = CognateDataset(data)
    assert len(dataset) == 9


