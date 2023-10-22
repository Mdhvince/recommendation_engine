import json
import configparser
from pathlib import Path
from typing import Dict

import torch

from src.user_item_indexer import UserItemIndexer


class Recommender:
    def __init__(self, config):
        self.cfg_data = config["DATA"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ui_indexer = UserItemIndexer(config)

        self.mean_ratings, self.user_factors, self.item_factors, self.user_bias, self.item_bias = self.load()
        print(f"\nRunning on {self.device}.\n")

    def load(self):
        root_dir = Path(__file__).parent.parent
        data_inference_dir = root_dir / "data_inference"
        user_factors_path = data_inference_dir / self.cfg_data.get("user_factors_filename")
        item_factors_path = data_inference_dir / self.cfg_data.get("item_factors_filename")
        user_bias_path = data_inference_dir / self.cfg_data.get("user_bias_filename")
        item_bias_path = data_inference_dir / self.cfg_data.get("item_bias_filename")
        stats_path = data_inference_dir / self.cfg_data.get("stats_filename")

        user_factors_mat: torch.FloatTensor = torch.load(user_factors_path, map_location=self.device)
        item_factors_mat: torch.FloatTensor = torch.load(item_factors_path, map_location=self.device)

        with open(user_bias_path, "r") as f:
            user_bias: Dict[str, float] = json.load(f)
        with open(item_bias_path, "r") as f:
            item_bias: Dict[str, float] = json.load(f)
        with open(stats_path, "r") as f:
            stats: Dict[str, float] = json.load(f)

        mean_ratings = stats["mean_rating"]

        return mean_ratings, user_factors_mat, item_factors_mat, user_bias, item_bias

    def predict(self, user: str, item: str) -> float:
        u_idx, i_idx = self.ui_indexer(user, item)
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
