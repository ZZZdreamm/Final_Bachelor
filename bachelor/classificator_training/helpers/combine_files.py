import torch
import os

CONFIG = {
    "file_1_path": "classificator_training/preprocessed_vectors/mega_clip_segformer_midas_dpt4.pth",
    "file_2_path": "classificator_training/preprocessed_vectors/mega_clip_segformer_midas_dpt6.pth",
    "output_path": "classificator_training/preprocessed_vectors/mega_clip_segformer_midas_dpt_final2.pth",
    "overwrite_duplicates": True
}

def combine_torch_files(config):
    """
    Loads two torch files specified in config, combines their keys,
    and saves to the output path.
    """
    f1_path = config["file_1_path"]
    f2_path = config["file_2_path"]
    out_path = config["output_path"]

    if not os.path.exists(f1_path):
        raise FileNotFoundError(f"File 1 not found: {f1_path}")
    if not os.path.exists(f2_path):
        raise FileNotFoundError(f"File 2 not found: {f2_path}")

    dict1 = torch.load(f1_path, map_location='cpu')

    dict2 = torch.load(f2_path, map_location='cpu')

    if not isinstance(dict1, dict) or not isinstance(dict2, dict):
        raise TypeError("Both files must contain dictionaries (not raw tensors or models).")

    common_keys = set(dict1.keys()) & set(dict2.keys())
    if common_keys:
        if not config["overwrite_duplicates"]:
            raise ValueError("Duplicate keys found and 'overwrite_duplicates' is False. Aborting.")

    combined_dict = {**dict1, **dict2}

    torch.save(combined_dict, out_path)

if __name__ == "__main__":
    if not os.path.exists("file1_test.pt"):
        torch.save({"layer1": torch.tensor([1.0]), "meta": "v1"}, "file1_test.pt")
        torch.save({"layer2": torch.tensor([2.0]), "meta": "v2"}, "file2_test.pt")
        CONFIG["file_1_path"] = "file1_test.pt"
        CONFIG["file_2_path"] = "file2_test.pt"
        CONFIG["output_path"] = "combined_test.pt"

    combine_torch_files(CONFIG)