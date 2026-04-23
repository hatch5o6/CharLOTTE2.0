import json
# from bli import BLIBasic
from sloth_hatch.sloth import read_json

def extract_cognates(
    src_hf_model_name,
    tgt_file,
    src_lang,
    tgt_lang,
    cognate_list_out,
    theta=0.5,
):
    word_list_path = cognate_list_out + ".WORD_LIST.json"
    bli_obj = BLIBasic(
        HF_MODEL_NAME=src_hf_model_name,
        TARGET_FILE_PATH=tgt_file,
        OUTPATH=word_list_path,
        batch_size=100,
        iterations=3,
        threshold=theta
    )
    # submitit bli_obj.main()

def get_cognate_list(in_path, theta):
    data = read_json(in_path)
    cognate_list = []
    for word, results in data.items():
        if word in ["batch_no", "batch_size"]: continue
        results = [(1 - dist, cognate) for cognate, dist in results.items()]
        results.sort(reverse=True)
        dist, cognate = results[0]
        if dist <= theta:
            cognate_list.append((word.strip(), cognate.strip(), dist))
    cognate_list.sort(reverse=True, key=lambda x: x[-1])
    return cognate_list

def write_cognates(cognate_list, out_path):
    with open(out_path, "w") as outf:
        for src_word, cognate, dist in cognate_list:
            outf.write(" ||| ".join(["N/A", src_word, cognate, str(dist)]) + "\n")

if __name__ == "__main__":
    for path in [
        "/home/hatch5o6/nobackup/archive/CharLOTTE2.0/test_bli_an/basic_noninit+hintokvocab.an.simfunc_ned.batchsize_100.iterations_3.threshold_0.5.json",
        "/home/hatch5o6/nobackup/archive/CharLOTTE2.0/test_bli_mfe/basic_noninit+hintokvocab.mfe.simfunc_ned.batchsize_100.iterations_3.threshold_0.5.json"
    ]:
        assert path.endswith('.json')
        out_path = path[:-4] + ".txt"
        cognate_list = get_cognate_list(path, 0.5)
        write_cognates(cognate_list, out_path)
