import yaml
import json
import argparse

def main(
    config_file,
    python_files,
    out_file
):
    config = read_yaml(config_file)
    found = {k: [] for k in config}
    for key in config.keys():
        for p in python_files:
            for l, line in enumerate(read_lines(p)):
                if f'"{key}"' in line:
                    found[key].append(str((l, p, line)))
    with open(out_file, "w") as outf:
        outf.write(json.dumps(found, indent=2))
    print("NOT FOUND:")
    for k, v in found.items():
        if len(v) == 0:
            print("\t", k)

def read_lines(f):
    with open(f) as inf:
        lines = [l.rstrip() for l in inf.readlines()]
    return lines

def read_yaml(f):
    with open(f) as inf:
        data = yaml.safe_load(inf)
    return data

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_file", default="/home/hatch5o6/CharLOTTE2.0/code/configs/test.yaml")
    parser.add_argument("-p", "--python_files", help="comma-delimited list of files to search", default="/home/hatch5o6/CharLOTTE2.0/code/OC/src/OC.py,/home/hatch5o6/CharLOTTE2.0/code/OC/src/OCLightning.py,/home/hatch5o6/CharLOTTE2.0/code/OC/src/train.py")
    parser.add_argument("-o", "--out", default="/home/hatch5o6/CharLOTTE2.0/code/OC/out/python/confirm_config_keys.out")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    python_files = [f.strip() for f in args.python_files.split(",")]
    main(args.config_file, python_files, args.out)
