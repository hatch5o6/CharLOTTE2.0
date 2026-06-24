import pytest
import os
import shutil
from sloth_hatch import sloth

from OC.reshape import reshape

########################
# prepare_source_words #
########################

def test_prepare_source_words_long_enough_0():
    output_dir = "/home/hatch5o6/CharLOTTE2.0/src/OC/reshape/tests/fixtures/outputs"
    sloth.create_directory(output_dir, destroy=True)
    assert os.listdir(output_dir) == []

    output_path = os.path.join(output_dir, "prepared_pl_words.txt")
    assert not os.path.exists(output_path)
    pl_files = [
        "src/OC/reshape/tests/fixtures/prepare_pl_files/pl1.txt",
        "src/OC/reshape/tests/fixtures/prepare_pl_files/pl2.txt",
        "src/OC/reshape/tests/fixtures/prepare_pl_files/pl3.txt"
    ]

    reshape.prepare_source_words(pl_files=pl_files,
                                 long_enough=0,
                                 out_path=output_path)
    
    assert sloth.read_lines(output_path) == sorted([
        "-1 ||| Hello ||| <N/A> ||| -1.0",
        "-1 ||| there ||| <N/A> ||| -1.0",
        "-1 ||| Is ||| <N/A> ||| -1.0",
        "-1 ||| anyone ||| <N/A> ||| -1.0",
        "-1 ||| home ||| <N/A> ||| -1.0",
        "-1 ||| There ||| <N/A> ||| -1.0",
        "-1 ||| were ||| <N/A> ||| -1.0",
        "-1 ||| 1,035 ||| <N/A> ||| -1.0",
        "-1 ||| men ||| <N/A> ||| -1.0",
        "-1 ||| in ||| <N/A> ||| -1.0",
        "-1 ||| the ||| <N/A> ||| -1.0",
        "-1 ||| camp ||| <N/A> ||| -1.0",
        "-1 ||| that ||| <N/A> ||| -1.0",
        "-1 ||| night ||| <N/A> ||| -1.0",
        "-1 ||| I'll ||| <N/A> ||| -1.0",
        "-1 ||| charge ||| <N/A> ||| -1.0",
        "-1 ||| you ||| <N/A> ||| -1.0",
        "-1 ||| 35.50 ||| <N/A> ||| -1.0",
        "-1 ||| for ||| <N/A> ||| -1.0",
        "-1 ||| glass ||| <N/A> ||| -1.0",
        "-1 ||| of ||| <N/A> ||| -1.0",
        "-1 ||| water ||| <N/A> ||| -1.0",
        "-1 ||| Follow ||| <N/A> ||| -1.0",
        "-1 ||| me ||| <N/A> ||| -1.0",
        "-1 ||| at ||| <N/A> ||| -1.0",
        "-1 ||| myhandle ||| <N/A> ||| -1.0",
        "-1 ||| on ||| <N/A> ||| -1.0",
        "-1 ||| X ||| <N/A> ||| -1.0",
        "-1 ||| The ||| <N/A> ||| -1.0",
        "-1 ||| child ||| <N/A> ||| -1.0",
        "-1 ||| eats ||| <N/A> ||| -1.0",
        "-1 ||| cookies ||| <N/A> ||| -1.0",
        "-1 ||| because ||| <N/A> ||| -1.0",
        "-1 ||| he ||| <N/A> ||| -1.0",
        "-1 ||| thinks ||| <N/A> ||| -1.0",
        "-1 ||| they're ||| <N/A> ||| -1.0",
        "-1 ||| delicious ||| <N/A> ||| -1.0",
        "-1 ||| But ||| <N/A> ||| -1.0",
        "-1 ||| an ||| <N/A> ||| -1.0",
        "-1 ||| adult ||| <N/A> ||| -1.0",
        "-1 ||| also ||| <N/A> ||| -1.0",
        "-1 ||| What ||| <N/A> ||| -1.0",
        "-1 ||| tarnation ||| <N/A> ||| -1.0",
        "-1 ||| heck ||| <N/A> ||| -1.0",
        "-1 ||| do ||| <N/A> ||| -1.0",
        "-1 ||| think ||| <N/A> ||| -1.0",
        "-1 ||| ye'r ||| <N/A> ||| -1.0",
        "-1 ||| doin ||| <N/A> ||| -1.0",
    ])

    shutil.rmtree(output_dir)


def test_prepare_source_words_long_enough_3():
    output_dir = "/home/hatch5o6/CharLOTTE2.0/src/OC/reshape/tests/fixtures/outputs"
    sloth.create_directory(output_dir, destroy=True)
    assert os.listdir(output_dir) == []

    output_path = os.path.join(output_dir, "prepared_pl_words.txt")
    assert not os.path.exists(output_path)
    pl_files = [
        "src/OC/reshape/tests/fixtures/prepare_pl_files/pl1.txt",
        "src/OC/reshape/tests/fixtures/prepare_pl_files/pl2.txt",
        "src/OC/reshape/tests/fixtures/prepare_pl_files/pl3.txt"
    ]

    reshape.prepare_source_words(pl_files=pl_files,
                                 long_enough=3,
                                 out_path=output_path)
    
    assert sloth.read_lines(output_path) == sorted([
        "-1 ||| Hello ||| <N/A> ||| -1.0",
        "-1 ||| there ||| <N/A> ||| -1.0",
        "-1 ||| anyone ||| <N/A> ||| -1.0",
        "-1 ||| home ||| <N/A> ||| -1.0",
        "-1 ||| There ||| <N/A> ||| -1.0",
        "-1 ||| were ||| <N/A> ||| -1.0",
        "-1 ||| 1,035 ||| <N/A> ||| -1.0",
        "-1 ||| men ||| <N/A> ||| -1.0",
        "-1 ||| the ||| <N/A> ||| -1.0",
        "-1 ||| camp ||| <N/A> ||| -1.0",
        "-1 ||| that ||| <N/A> ||| -1.0",
        "-1 ||| night ||| <N/A> ||| -1.0",
        "-1 ||| I'll ||| <N/A> ||| -1.0",
        "-1 ||| charge ||| <N/A> ||| -1.0",
        "-1 ||| you ||| <N/A> ||| -1.0",
        "-1 ||| 35.50 ||| <N/A> ||| -1.0",
        "-1 ||| for ||| <N/A> ||| -1.0",
        "-1 ||| glass ||| <N/A> ||| -1.0",
        "-1 ||| water ||| <N/A> ||| -1.0",
        "-1 ||| Follow ||| <N/A> ||| -1.0",
        "-1 ||| myhandle ||| <N/A> ||| -1.0",
        "-1 ||| The ||| <N/A> ||| -1.0",
        "-1 ||| child ||| <N/A> ||| -1.0",
        "-1 ||| eats ||| <N/A> ||| -1.0",
        "-1 ||| cookies ||| <N/A> ||| -1.0",
        "-1 ||| because ||| <N/A> ||| -1.0",
        "-1 ||| thinks ||| <N/A> ||| -1.0",
        "-1 ||| they're ||| <N/A> ||| -1.0",
        "-1 ||| delicious ||| <N/A> ||| -1.0",
        "-1 ||| But ||| <N/A> ||| -1.0",
        "-1 ||| adult ||| <N/A> ||| -1.0",
        "-1 ||| also ||| <N/A> ||| -1.0",
        "-1 ||| What ||| <N/A> ||| -1.0",
        "-1 ||| tarnation ||| <N/A> ||| -1.0",
        "-1 ||| heck ||| <N/A> ||| -1.0",
        "-1 ||| think ||| <N/A> ||| -1.0",
        "-1 ||| ye'r ||| <N/A> ||| -1.0",
        "-1 ||| doin ||| <N/A> ||| -1.0",
    ])

    shutil.rmtree(output_dir)



################
# reshape_data #
################
    
WORD_MAPPINGS = {
    "Hello": "Hecko",
    "there": "zer",
    "Is": "Iz",
    "anyone": "eeniwan",
    "home": "'ome",
    "There": "Zer",
    "were": "ver",
    "1,035": "1.035",
    "in": "een",
    "the": "da",
    "camp": "kamp",
    "that": "zat",
    "night": "nayt",
    "I'll": "eel",
    "charge": "chahj",
    "you": "iu",
    "35.50": "35,000,000",
    "for": "4",
    "glass": "goblet",
    "of": "uv",
    "water": "vasser",
    "me": "mii",
    "on": "ahnn",
    "X": "twitter",
    "The": "Da",
    "child": "niño",
    "eats": "eetz",
    "cookies": "biscuits",
    "because": "cuz",
    "he": "heee",
    "thinks": "sinks",
    "they're": "theeeeeey're",
    "delicious": "delicioso",
    "But": "However,",
    "an": "anne",
    "adult": "uhdolt",
    "also": "all-so"
}

def test_reshape_data_long_enough_0():
    pl_file = "src/OC/reshape/tests/fixtures/data_to_reshape/pl.txt"
    output_tag = ".reshaped"
    output_file = pl_file + output_tag
    if os.path.exists(output_file):
        os.remove(output_file)
    assert not os.path.exists(output_file)

    reshape.reshape_data(
        pl_file=pl_file,
        word_mappings=WORD_MAPPINGS,
        output_tag=output_tag,
        long_enough=0
    )

    assert sloth.read_lines(output_file) == [
        "Hecko, zer! Iz eeniwan 'ome??",
        "Zer ver 1.035 men een da kamp zat nayt.",
        "eel chahj iu $35,000,000 4 zat goblet uv vasser!!",
        "Follow mii at @myhandle ahnn twitter.",
        "Da niño eetz biscuits cuz heee sinks theeeeeey're delicioso.",
        "However, anne uhdolt eetz biscuits ...cuz... heee all-so sinks theeeeeey're delicioso."
    ]

    os.remove(output_file)


def test_reshape_data_long_enough_4():
    pl_file = "src/OC/reshape/tests/fixtures/data_to_reshape/pl.txt"
    output_tag = ".reshaped"
    output_file = pl_file + output_tag
    if os.path.exists(output_file):
        os.remove(output_file)
    assert not os.path.exists(output_file)

    reshape.reshape_data(
        pl_file=pl_file,
        word_mappings=WORD_MAPPINGS,
        output_tag=output_tag,
        long_enough=4
    )

    assert sloth.read_lines(output_file) == [
        "Hecko, zer! Is eeniwan 'ome??",
        "Zer ver 1.035 men in the kamp zat nayt.",
        "eel chahj you $35,000,000 for zat goblet of vasser!!",
        "Follow me at @myhandle on X.",
        "The niño eetz biscuits cuz he sinks theeeeeey're delicioso.",
        "But an uhdolt eetz biscuits ...cuz... he all-so sinks theeeeeey're delicioso."
    ]

    os.remove(output_file)
    