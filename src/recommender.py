import json
import configparser
from pathlib import Path
from typing import Dict

import torch
import numpy as np


class Recommender:
    def __init__(self, config):
        self.cfg_data = config["DATA"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.unseen_mode_nan = self.cfg_data.get("unseen_mode") == "nan"

        ratings_mat, self.user_factors, self.item_factors, self.user_bias, self.item_bias = self.load()
        seen_data = ~torch.isnan(ratings_mat) if self.unseen_mode_nan else ratings_mat != 0
        n_ratings = torch.sum(seen_data)
        self.mean_ratings = torch.sum(torch.where(torch.isnan(ratings_mat), 0., ratings_mat)) / n_ratings

        print(f"\nRunning on {self.device}.\n")

    def load(self):
        root_dir = Path(__file__).parent.parent
        data_dir = root_dir / "data"
        data_inference_dir = root_dir / "data_inference"
        ratings_path = data_dir / self.cfg_data.get("ratings_filename")
        user_factors_path = data_inference_dir / self.cfg_data.get("user_factors_filename")
        item_factors_path = data_inference_dir / self.cfg_data.get("item_factors_filename")
        user_bias_path = data_inference_dir / self.cfg_data.get("user_bias_filename")
        item_bias_path = data_inference_dir / self.cfg_data.get("item_bias_filename")

        ratings_mat: np.ndarray = np.load(ratings_path)
        ratings_mat: torch.FloatTensor = torch.from_numpy(ratings_mat).to(self.device)
        user_factors_mat: torch.FloatTensor = torch.load(user_factors_path, map_location=self.device)
        item_factors_mat: torch.FloatTensor = torch.load(item_factors_path, map_location=self.device)
        user_bias: Dict = json.load(open(user_bias_path))
        item_bias: Dict = json.load(open(item_bias_path))

        return ratings_mat, user_factors_mat, item_factors_mat, user_bias, item_bias

    def predict(self, u_idx: int, i_idx: int) -> float:
        predicted = torch.dot(self.user_factors[u_idx, :], self.item_factors[:, i_idx])
        bias = self.mean_ratings + self.user_bias[str(u_idx)] + self.item_bias[str(i_idx)]
        predicted += bias
        return predicted.item()


if __name__ == "__main__":
    config = configparser.ConfigParser(inline_comment_prefixes="#")
    config.read(Path(__file__).parent.parent / "config.ini")

    recommender = Recommender(config)
    rating = recommender.predict(0, 0)
    print(f"Rating: {rating:.2f}")
