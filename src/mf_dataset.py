import configparser
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class MFDataset(Dataset):
    def __init__(self, config, mode="train"):
        assert mode in ["train", "val"], f"mode must be 'train' or 'val', got {mode} instead."
        self.cfd_data = config["DATA"]
        self.unseen_mode = self.cfd_data.get("unseen_mode")
        data_dir = Path(__file__).parent.parent / Path(self.cfd_data.get("data_dirname"))

        if mode == "train":
            ratings_path = data_dir / Path(self.cfd_data.get("ratings_train_filename"))
        else:
            ratings_path = data_dir / Path(self.cfd_data.get("ratings_val_filename"))

        self.ratings_mat = MFDataset.load_ratings_matrix(ratings_path)


    def __len__(self):
        seen_data = ~np.isnan(self.ratings_mat) if self.unseen_mode == "nan" else self.ratings_mat != 0
        return np.sum(seen_data)


    def __getitem__(self, idx):
        seen_data = ~np.isnan(self.ratings_mat) if self.unseen_mode == "nan" else self.ratings_mat != 0
        seen_indices = np.where(seen_data)  # returns a tuple of arrays of indices for each dimension user/item
        user_indices = seen_indices[0]
        item_indices = seen_indices[1]
        user_idx = user_indices[idx]
        item_idx = item_indices[idx]
        rating = self.ratings_mat[user_idx, item_idx]
        return {
            "user_idx": torch.tensor(user_idx, dtype=torch.long),
            "item_idx": torch.tensor(item_idx, dtype=torch.long),
            "rating": torch.tensor(rating, dtype=torch.float32)
        }


    @staticmethod
    def save_train_val_matrices(config):
        cfd_data = config["DATA"]
        train_ratio = cfd_data.getfloat("train_ratio")
        unseen_mode = cfd_data.get("unseen_mode")
        root_dir = Path(__file__).parent.parent
        ratings_file = Path(cfd_data.get("data_dirname")) / cfd_data.get("ratings_filename")
        ratings_path = root_dir / ratings_file
        ratings_mat = MFDataset.load_ratings_matrix(ratings_path)
        train_mat, val_mat = MFDataset.split(ratings_mat, train_ratio, unseen_mode)
        train_path = root_dir / Path(cfd_data.get("data_dirname")) / Path(cfd_data.get("ratings_train_filename"))
        val_path = root_dir / Path(cfd_data.get("data_dirname")) / Path(cfd_data.get("ratings_val_filename"))
        np.save(train_path, train_mat)
        np.save(val_path, val_mat)
        print(f"Saved train matrix to {train_path}\nSaved val matrix to {val_path}.\n")


    @staticmethod
    def load_ratings_matrix(path):
        assert isinstance(path, Path), "path must be a pathlib.Path object"
        assert path.exists(), f"path {path} does not exist"
        return np.load(path)


    @staticmethod
    def split(matrix, train_ratio, unseen_mode="zero", seed=42):
        """
        Split a user-item matrix into a train and a test matrix. This is a particular (vertical) split, where the train
        matrix contains a random subset of the ratings of each user, and the test matrix contains the remaining ratings.

        Example: For a user with ratings [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], the train matrix could be:
        [1, 0, 0, 0, 5, 6, 0, 0, 0, 10]
        and the test matrix:
        [0, 2, 3, 4, 0, 0, 7, 8, 9, 0]

        :param matrix: ratings matrix
        :param train_ratio: ratio of ratings to put in the train matrix
        :param unseen_mode: whether to put unseen ratings as 0 or NaN
        :param seed: random seed
        :return: train matrix, test matrix
        """
        assert isinstance(matrix, np.ndarray), "matrix must be a numpy array"
        assert unseen_mode in ["zero", "nan"], "unseen_mode must be 'zero' or 'nan', got {unseen_mode} instead"
        np.random.seed(seed)

        train_matrix = np.zeros_like(matrix) if unseen_mode == "zero" else np.nan * np.zeros_like(matrix)
        test_matrix = np.zeros_like(matrix) if unseen_mode == "zero" else np.nan * np.zeros_like(matrix)

        for user_index in range(matrix.shape[0]):
            user_row = matrix[user_index, :]
            rated_indices = np.where(user_row != 0)[0] if unseen_mode == "zero" else np.where(~np.isnan(user_row))[0]
            n_rated = len(rated_indices)
            train_size = int(n_rated * train_ratio)
            train_indices = np.random.choice(rated_indices, size=train_size, replace=False)
            test_indices = np.setdiff1d(rated_indices, train_indices)
            train_matrix[user_index, train_indices] = user_row[train_indices]
            test_matrix[user_index, test_indices] = user_row[test_indices]

        return train_matrix, test_matrix


if __name__ == "__main__":
    config = configparser.ConfigParser(inline_comment_prefixes="#")
    config.read(Path(__file__).parent.parent / "config.ini")
    MFDataset.save_train_val_matrices(config)



    # define the loss function and the optimizer


