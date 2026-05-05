import argparse
import os
import csv
from tqdm import tqdm
import re

def main(DATA):
    for d in os.listdir(DATA):
        d_path = os.path.join(DATA, d)
        assert os.path.isdir(d_path)
        print("Extracting", d_path)

        pairs = []

        transcripts_d = os.path.join(d_path, "transcripts")
        translations_d = os.path.join(d_path, "translations")
        
        assert len(os.listdir(transcripts_d)) == len(os.listdir(translations_d))
        for transcript_f in tqdm(os.listdir(transcripts_d)):
            assert transcript_f.endswith(".tsv")
            translation_f = transcript_f[:-4] + ".eng.tsv"

            transcript_path = os.path.join(transcripts_d, transcript_f)
            translation_path = os.path.join(translations_d, translation_f)

            transcript = read_tsv(transcript_path)
            translation = read_tsv(translation_path)

            # I'm assuming that if there is a difference in the number of segments that the translation is the shorter one, because maybe some segment in the orginal transcript wasn't translated
            assert len(translation) <= len(transcript)
            if len(transcript) != len(translation):
                print("transcript_path:", transcript_path)
                print(f"\tlen transcript: {len(transcript)}")
                print("translation_path:", translation_path)
                print(f"\tlen translation: {len(translation)}")
            # assert len(transcript) == len(translation)
            for (start, end), transcript_text in transcript.items():
                if (start, end) in translation:
                    translation_text = translation[(start, end)]
                    transcript_text = clean_transcription(transcript_text)
                    translation_text = clean_translation(translation_text)
                    pairs.append((transcript_text, translation_text))

        extracted_folder = os.path.join(d_path, "extracted_parallel")
        if not os.path.exists(extracted_folder):
            os.mkdir(extracted_folder)
        src_f = os.path.join(extracted_folder, "all.aeb.txt")
        tgt_f = os.path.join(extracted_folder, "all.eng.txt")

        with open(src_f, "w") as sf, open(tgt_f, "w") as tf:
            for aeb_sent, eng_sent in pairs:
                sf.write(aeb_sent.strip() + "\n")
                tf.write(eng_sent.strip() + "\n")

def clean_transcription(sent):
    # see README.txt. Removing MSA, foreign, uncertain, uncertain + MSA, uncertain + foreign tags
    sent = sent.replace("UM/", "") \
        .replace("UO/", "") \
        .replace("M/", "") \
        .replace("O/", "") \
        .replace("U/", "") 
    # normalize whitespace
    sent = re.sub(r'\s+', ' ', sent).strip()
    # normalize spacing before punctuation
    # DECIDED NOT TO NORMALIZE PUNCT SPACING
    # sent = normalize_punct_spacing(sent)
    # normalize whitespace again
    sent = re.sub(r'\s+', ' ', sent).strip()
    return sent

def clean_translation(sent):
    # See README.txt. Removing partial word, foreign word, mispronounced word, typographical error tags. 
    # Also remove double (()) paranthese which mark uncertain word or words.
    sent = sent.replace("%pw", "") \
        .replace("#", "") \
        .replace("+", "") \
        .replace("=", "") \
        .replace("((", "") \
        .replace("))", "")
    # normalize whitespace
    sent = re.sub(r'\s+', ' ', sent).strip()
    # normalize spacing before punctuation
    # DECIDED NOT TO NORMALIZE PUNCT SPACING
    # sent = normalize_punct_spacing(sent)
    # normalize whitespace again
    sent = re.sub(r'\s+', ' ', sent).strip()
    return sent

def normalize_punct_spacing(sent):
    sent = re.sub(r"\s+,", ",", sent)
    sent = re.sub(r"\s+\.", ".", sent)
    sent = re.sub(r"\s+;", ";", sent)
    sent = re.sub(r"\s+\?", "?", sent)
    sent = re.sub(r"\s+؛", "؛", sent)
    sent = re.sub(r"\s+:", ":", sent)
    sent = re.sub(r"\s+،", "،", sent)
    sent = re.sub(r"\s+؟", "؟", sent)
    sent = re.sub(r"\s+!", "!", sent)
    return sent

def read_tsv(f):
    with open(f, newline='') as inf:
        tsv_reader = csv.reader(inf, delimiter="\t")
        data = [tuple(r) for r in tsv_reader]
    data_dict = {}
    for start, end, duration, sent in data:
        if (start, end) in data_dict:
            print(f"START-END {(start, end)} already in data_dict")
            print("\t-->", f)
            print("\twill replace annotation with this one:")
            print("\t\t", f"`{data_dict[(start, end)]}` -> `{sent}`")
        # assert (start, end) not in data_dict
        # we will just replace it instead, always taking the last annotation for the segment if there's multiple annotations
        data_dict[(start, end)] = sent
    return data_dict

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", 
                        "-d",
                        default="example/path/to/IWSLT22_and_IWSLT23_Tunisian_Arabic_Shared_Task/data",
                        help="path to IWSLT22_and_IWSLT23_Tunisian_Arabic_Shared_Task/data directory")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    main(args.data_folder)
