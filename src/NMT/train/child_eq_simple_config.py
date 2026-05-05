import argparse
from sloth_hatch.sloth import read_yaml

def main(yaml_f):
    print(f"Checking if child and simple configs match in:\n\t`{yaml_f}`")
    config = read_yaml(yaml_f)
    ct_failed = 0
    for k, v in config.items():
        if k.startswith("child_"):
            simple_k = "simple" + k[5:]
            print(f"{k} vs {simple_k}:")
            # print(f"\t{v} vs {config[simple_k]}")
            if v == config[simple_k]:
                pass
            else:
                print(f"\t{v} vs {config[simple_k]}")
                print("\tfail")
                ct_failed += 1
    print(f"{ct_failed} failed.")
    if ct_failed == 0:
        print("ALL PASSED :)")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml", "-y")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    main(args.yaml)