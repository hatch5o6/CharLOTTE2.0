from datasets import load_dataset, get_dataset_config_names
# from https://huggingface.co/datasets/prajdabre/KreolMorisienMT

configs = get_dataset_config_names("prajdabre/KreolMorisienMT")
print(configs)
dataset = load_dataset("prajdabre/KreolMorisienMT")


# for lang_pair in ["cr", "en-cr", "fr-cr"]:
#     for div in ["train", "validation", "test"]:
#         dataset = load_dataset("prajdabre/KreolMorisienMT", lang_pair, split=div)
#         print(dataset)
