import os
from sloth_hatch.sloth import read_lines, write_lines

map_lang = {
    "es": "xx",
    "an": "yy",
    "en": "zz",

    "fr": "vv",
    "mfe": "ww"
}
charlotte_data="/home/hatch5o6/groups/grp_charlotte/nobackup/archive/char2.0_data/CharLOTTE_data"

def copy_samples(src_dir, tgt_dir, size=1000):
    os.makedirs(tgt_dir, exist_ok=True)
    for f in os.listdir(src_dir):
        f_path = os.path.join(src_dir, f)
        lines = read_lines(f_path)[:size]

        lang = f.split(".")[-2]
        newlang = map_lang[lang]
        new_f = f.replace(f".{lang}.", f".{newlang}.")
        new_path = os.path.join(tgt_dir, new_f)

        write_lines(lines, new_path)

for pl, cl, tl in [
    ("es", "an", "en"),
    ("fr", "mfe", "en")
]:
    pl_cl = os.path.join(charlotte_data, f"{pl}-{cl}")
    pl_tl = os.path.join(charlotte_data, f"{pl}-{tl}")
    cl_tl = os.path.join(charlotte_data, f"{cl}-{tl}")

    target_pl_cl = os.path.join(charlotte_data, f"{map_lang[pl]}-{map_lang[cl]}")
    target_pl_tl = os.path.join(charlotte_data, f"{map_lang[pl]}-{map_lang[tl]}")
    target_cl_tl = os.path.join(charlotte_data, f"{map_lang[cl]}-{map_lang[tl]}")

    copy_samples(pl_cl, target_pl_cl)
    copy_samples(pl_tl, target_pl_tl)
    copy_samples(cl_tl, target_cl_tl)

