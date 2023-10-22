import json
import configparser
import pickle
from collections import namedtuple
from pathlib import Path
from typing import Dict

import torch
import numpy as np


class Recommender:
    def __init__(self, config):
        self.cfg_data = config["DATA"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.unseen_mode_nan = self.cfg_data.get("unseen_mode") == "nan"

        (self.mean_ratings,
         self.user_factors, self.item_factors,
         self.user_bias, self.item_bias,
         self.u_name_to_idx, self.it_name_to_idx) = self.load()

        print(f"\nRunning on {self.device}.\n")

    def load(self):
        root_dir = Path(__file__).parent.parent
        data_inference_dir = root_dir / "data_inference"
        user_factors_path = data_inference_dir / self.cfg_data.get("user_factors_filename")
        item_factors_path = data_inference_dir / self.cfg_data.get("item_factors_filename")
        user_bias_path = data_inference_dir / self.cfg_data.get("user_bias_filename")
        item_bias_path = data_inference_dir / self.cfg_data.get("item_bias_filename")
        stats_path = data_inference_dir / self.cfg_data.get("stats_filename")
        interactions_path = data_inference_dir / self.cfg_data.get("interactions_filename")

        user_factors_mat: torch.FloatTensor = torch.load(user_factors_path, map_location=self.device)
        item_factors_mat: torch.FloatTensor = torch.load(item_factors_path, map_location=self.device)

        with open(user_bias_path, "r") as f:
            user_bias: Dict[str, float] = json.load(f)
        with open(item_bias_path, "r") as f:
            item_bias: Dict[str, float] = json.load(f)
        with open(stats_path, "r") as f:
            stats: Dict[str, float] = json.load(f)
        with open(interactions_path, "rb") as f:
            interactions_dict: Dict[str, float] = pickle.load(f)

        mean_ratings = stats["mean_rating"]
        u_name_to_idx = {eval(k)[2]: int(eval(k)[0]) for k, _ in interactions_dict.items()}
        it_name_to_idx = {eval(k)[3]: int(eval(k)[1]) for k, _ in interactions_dict.items()}

        return mean_ratings, user_factors_mat, item_factors_mat, user_bias, item_bias, u_name_to_idx, it_name_to_idx

    def predict(self, user: str, item: str) -> float:
        u_idx, i_idx = self.u_name_to_idx[user], self.it_name_to_idx[item]
        predicted = torch.dot(self.user_factors[u_idx, :], self.item_factors[:, i_idx])
        bias = self.mean_ratings + self.user_bias[str(u_idx)] + self.item_bias[str(i_idx)]
        predicted += bias
        return predicted.item()


if __name__ == "__main__":
    config = configparser.ConfigParser(inline_comment_prefixes="#")
    config.read(Path(__file__).parent.parent / "config.ini")

    recommender = Recommender(config)

    user_name = "62990992"
    item_name = "Counter-Strike"
    rating = recommender.predict(user_name, item_name)
    print(f"Rating: {rating:.2f}")
