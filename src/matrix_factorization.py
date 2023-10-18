import configparser
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np



class MatrixFactorization:
    def __init__(self, config):
        """
        :param config: config object
        """
        self.cfg_data = config["DATA"]
        self.cfg_train = config["TRAINING"]
        self.cfg_train.getint("latent_factors")
        self.unseen_mode_nan = self.cfg_data.get("unseen_mode") == "nan"
        self.min_rating, self.max_rating = self.cfg_data.getfloat("min_rating"), self.cfg_data.getfloat("max_rating")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.l2 = self.cfg_train.getfloat("l2_reg")

        self.ratings_mat, self.train_mat, self.val_mat = self.init_matrices()
        self.n_users, self.n_items, self.n_train_ratings, self.n_val_ratings, self.ratings_avg = self.matrices_info()
        self.user_factors, self.item_factors = self.init_learnable_factors()
        self.lrs = self.learning_rate_decay()

        self.nic = 0  # no improvement counter
        self.best_loss = float("Inf")
        self.user_bias = defaultdict(lambda: 0)
        self.item_bias = defaultdict(lambda: 0)
        self.best_user_factors, self.best_item_factors = None, None



    def init_matrices(self):
        root_dir = Path(__file__).parent.parent
        ratings_file = Path(self.cfg_data.get("data_dirname")) / self.cfg_data.get("ratings_filename")
        ratings_mat = MatrixFactorization.load_ratings_matrix(root_dir / ratings_file)
        assert isinstance(ratings_mat, np.ndarray), "user_item must be a numpy array"

        train_mat, val_mat = MatrixFactorization.split(
            ratings_mat,
            train_ratio=self.cfg_data.getfloat("train_ratio"),
            unseen_mode=self.cfg_data.get("unseen_mode")
        )
        return torch.tensor(ratings_mat, dtype=torch.float32, device=self.device), \
                  torch.tensor(train_mat, dtype=torch.float32, device=self.device), \
                    torch.tensor(val_mat, dtype=torch.float32, device=self.device)


    def init_learnable_factors(self):
        user_factors = np.random.rand(self.n_users, self.cfg_train.getint("latent_factors"))
        item_factors = np.random.rand(self.cfg_train.getint("latent_factors"), self.n_items)
        return torch.tensor(user_factors, dtype=torch.float32, device=self.device), \
                    torch.tensor(item_factors, dtype=torch.float32, device=self.device)


    def matrices_info(self):
        n_users, n_items = self.ratings_mat.shape
        seen_data = ~torch.isnan(self.ratings_mat) if self.unseen_mode_nan else self.ratings_mat != 0
        seen_train = ~torch.isnan(self.train_mat) if self.unseen_mode_nan else self.train_mat != 0
        seen_val = ~torch.isnan(self.val_mat) if self.unseen_mode_nan else self.val_mat != 0

        # count the number of ratings and the average rating
        n_ratings = torch.sum(seen_data)
        n_ratings_train = torch.sum(seen_train)
        n_ratings_val = torch.sum(seen_val)
        mean_ratings = torch.sum(torch.where(torch.isnan(self.ratings_mat), 0., self.ratings_mat)) / n_ratings
        return n_users, n_items, n_ratings_train, n_ratings_val, mean_ratings


    def results_generator(self, rating_mat):
        """
        Generator to iterate over user-item interactions
        """
        for user_index in range(self.n_users):
            for item_index in range(self.n_items):
                actual_rating = rating_mat[user_index, item_index]
                is_unseen = torch.isnan(actual_rating) if self.unseen_mode_nan else actual_rating == 0
                if is_unseen: continue
                predicted_rating = self.predict(user_index, item_index)
                error = actual_rating - predicted_rating
                yield user_index, item_index, actual_rating, predicted_rating, error


    def predict(self, u_idx, i_idx):
        predicted = torch.dot(self.user_factors[u_idx, :], self.item_factors[:, i_idx])
        bias = self.ratings_avg + self.user_bias[u_idx] + self.item_bias[i_idx]
        predicted += bias
        # predicted = np.clip(predicted, self.min_rating, self.max_rating)
        return predicted


    def optimize(self, epoch, user_index, item_index, error):
        """
        Update the user and item factors using SGD
        :param epoch:
        :param user_index:
        :param item_index:
        :param error:
        :return:
        """
        momentum = .9
        user_factors_momentum = 0
        item_factors_momentum = 0

        for k in range(self.cfg_train.getint("latent_factors")):
            user_factor_gradient = error * self.item_factors[k, item_index] - self.l2 * self.user_factors[user_index, k]
            item_factor_gradient = error * self.user_factors[user_index, k] - self.l2 * self.item_factors[k, item_index]

            user_factors_momentum = momentum * user_factors_momentum + self.lrs[epoch] * user_factor_gradient
            item_factors_momentum = momentum * item_factors_momentum + self.lrs[epoch] * item_factor_gradient

            self.user_factors[user_index, k] += user_factors_momentum
            self.item_factors[k, item_index] += item_factors_momentum


    def train(self, epoch):
        """
        Learn the latent factors and biases using SGD
        :param epoch: current epoch
        :return: updated user_factors, item_factors, mse train loss
        """
        running_sse_train = 0
        for results in self.results_generator(self.train_mat):
            user_index, item_index, actual_rating, predicted_rating, error = results
            running_sse_train += error ** 2
            self.update_biases(user_index, item_index, error)
            self.optimize(epoch, user_index, item_index, error)

        return running_sse_train / self.n_train_ratings


    def evaluate(self):
        """
        Evaluate the model on the validation set
        :return: mse val loss
        """
        running_sse_val = 0
        for results in self.results_generator(self.val_mat):
            user_index, item_index, actual_rating, predicted_rating, error = results
            running_sse_val += error ** 2

        return running_sse_val / self.n_val_ratings


    def learn(self):
        self.best_loss = float("Inf")
        self.nic = 0

        for epoch in range(self.cfg_train.getint("num_epochs")):
            mse_train = self.train(epoch)
            mse_val = self.evaluate()
            if self.can_stop(epoch, mse_val): break
            self.log(epoch, mse_train, mse_val)


    def log(self, epoch, mse_train, mse_val):
        print(
            f"Epoch: {epoch + 1}/{self.cfg_train.getint('num_epochs')} "
            f"| Train RMSE: {np.sqrt(mse_train):.3f} | Val RMSE: {np.sqrt(mse_val):.3f}"
        )


    def can_stop(self, epoch, mse_val):
        """
        Check if the model can stop learning and update the best factors if needed
        :param epoch:
        :param mse_val:
        :return: True if the model can stop learning, False otherwise
        """
        if mse_val < self.best_loss and epoch > (self.cfg_train.getint("num_epochs") * 0.1):
            self.best_loss = mse_val
            self.best_user_factors = self.user_factors
            self.best_item_factors = self.item_factors
            self.nic = 0  # no improvement counter
        else:
            if epoch > (self.cfg_train.getint("num_epochs") * 0.1):
                self.nic += 1
        
        if self.cfg_train.getboolean("early_stopping") and (self.nic >= self.cfg_train.getint("patience")):
            return True
        return False


    def update_biases(self, user_index, item_index, error):
        self.user_bias[user_index] += self.cfg_train.getfloat("bias_lr") \
                                    * (error - self.cfg_train.getfloat("bias_reg") * self.user_bias[user_index])
        self.item_bias[item_index] += self.cfg_train.getfloat("bias_lr") \
                                    * (error - self.cfg_train.getfloat("bias_reg") * self.item_bias[item_index])


    def learning_rate_decay(self):
        learning_rates = []
        for epoch in range(self.cfg_train.getint("num_epochs")):
            lr = self.cfg_train.getfloat("end_lr") \
                 + (self.cfg_train.getfloat("start_lr") - self.cfg_train.getfloat("end_lr")) \
                 * np.exp(-self.cfg_train.getfloat("decay_rate") * epoch)
            learning_rates.append(lr)
        return learning_rates


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


    @staticmethod
    def load_ratings_matrix(path):
        assert isinstance(path, Path), "path must be a pathlib.Path object"
        assert path.exists(), f"path {path} does not exist"
        return np.load(path)



if __name__ == "__main__":
    config = configparser.ConfigParser(inline_comment_prefixes="#")
    config.read(Path(__file__).parent.parent / "config.ini")

    mf = MatrixFactorization(config)
    mf.learn()


