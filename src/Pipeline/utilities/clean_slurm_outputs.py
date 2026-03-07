import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--top", "-t", type=int, default=1)
args = parser.parse_args()
top = args.top

dirs = [
    "/home/hatch5o6/CharLOTTE2.0/code/Pipeline/out/slurm",
    "/home/hatch5o6/CharLOTTE2.0/code/NMT/out/slurm",
    "/home/hatch5o6/CharLOTTE2.0/code/OC/out/slurm"
]
for dir in dirs:
    if not os.path.exists(dir): continue
    fs = os.listdir(dir)
    files = {}
    for f in fs:        
        if os.path.isdir(os.path.join(dir, f)): continue
        number = f.split("_")[0]
        number = int(number)
        name = "_".join(f.split("_")[1:])
        if name not in files:
            files[name] = []
        
        files[name].append((number, f))

    for name, fs in files.items():
        fs.sort(reverse=True)

        for n, f in fs[top:]:
            f_path = os.path.join(dir, f)
            os.remove(f_path)

# clean core dumps
base_dir = "/home/hatch5o6/CharLOTTE2.0/code"
for f in os.listdir(base_dir):
    if f.startswith("core."):
        f_path = os.path.join(base_dir, f)
        print("removing", f_path)
        os.remove(f_path)