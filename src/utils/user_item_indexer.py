import pickle
from pathlib import Path
from typing import Dict


class UserItemIndexer:
    def __init__(self, config):
        self.cfg = config["DATA"]
        self.interactions_dict = self.load()
        self.user_name_to_idx = {eval(k)[2]: int(eval(k)[0]) for k, _ in self.interactions_dict.items()}
        self.item_name_to_idx = {eval(k)[3]: int(eval(k)[1]) for k, _ in self.interactions_dict.items()}

    def load(self):
        root_dir = Path(__file__).parent.parent
        data_inference_dir = root_dir / "data_inference"
        interactions_path = data_inference_dir / self.cfg.get("interactions_filename")

        with open(interactions_path, "rb") as f:
            interactions_dict: Dict[str, float] = pickle.load(f)

        return interactions_dict

    def split(self):
        """
        At training time
        """
        interactions_train = {}
        interactions_val = {}

        for key, rating in self.interactions_dict.items():
            user_index, item_index, _, _, split = eval(key)
            if split == "train":
                interactions_train[f"{(user_index, item_index)}"] = rating
            elif split == "val":
                interactions_val[f"{(user_index, item_index)}"] = rating
            else:
                raise ValueError(f"split value {split} is invalid")

        return interactions_train, interactions_val


    def __call__(self, user: str, item: str):
        """
        At inference time
        """
        try:
            u_idx = self.user_name_to_idx[user]
            i_idx = self.item_name_to_idx[item]
        except KeyError:
            print(f"User {user} or item {item} not found in dataset")
            return None, None
        return u_idx, i_idx
