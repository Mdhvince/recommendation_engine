import configparser
from pathlib import Path
from collections import defaultdict

import numpy as np



class MatrixFactorization:
    def __init__(self, config):
        """
        :param config: config object
        """
        self.cfg_dft = config["DEFAULT"]
        self.cfg_train = config["TRAINING"]
        self.cfg_train.getint("latent_factors")

        self.ratings_mat, self.train_mat, self.val_mat = self.init_matrices()
        self.n_users, self.n_items, self.n_train_ratings, self.n_val_ratings, self.ratings_avg = self.get_matrices_info()
        self.user_factors, self.item_factors = self.init_learnable_factors()
        self.lrs = self.learning_rate_decay()

        self.nic = 0  # no improvement counter
        self.best_loss = float("Inf")
        self.user_bias = defaultdict(lambda: 0)
        self.item_bias = defaultdict(lambda: 0)
        self.best_user_factors, self.best_item_factors = None, None


    def init_matrices(self):
        root_dir = Path(__file__).parent.parent
        ratings_file = Path(self.cfg_dft.get("data_dirname")) / self.cfg_dft.get("ratings_filename")
        ratings_mat = MatrixFactorization.load_ratings_matrix(root_dir / ratings_file)
        assert isinstance(ratings_mat, np.ndarray), "user_item must be a numpy array"

        train_mat, val_mat = MatrixFactorization.split(
            ratings_mat, train_ratio=self.cfg_train.getfloat("train_ratio"), unseen_mode="nan"
        )
        return ratings_mat, train_mat, val_mat


    def get_matrices_info(self):
        n_users, n_items = self.ratings_mat.shape
        num_ratings = np.sum(~np.isnan(self.ratings_mat))
        num_ratings_train = np.sum(~np.isnan(self.train_mat))
        num_ratings_val = np.sum(~np.isnan(self.val_mat))
        mean_ratings = np.sum(~np.isnan(self.ratings_mat)) / num_ratings
        return n_users, n_items, num_ratings_train, num_ratings_val, mean_ratings


    def predict(self, u_idx, i_idx):
        predicted = np.dot(self.user_factors[u_idx, :], self.item_factors[:, i_idx])
        bias = self.ratings_avg + self.user_bias[u_idx] + self.item_bias[i_idx]
        predicted += bias
        return predicted


    def init_learnable_factors(self):
        user_factors = np.random.rand(self.n_users, self.cfg_train.getint("latent_factors"))
        item_factors = np.random.rand(self.cfg_train.getint("latent_factors"), self.n_items)
        return user_factors, item_factors


    def train(self, epoch):
        """
        Learn the latent factors and biases using SGD
        :param epoch: current epoch
        :return: updated user_factors, item_factors, mse train loss
        """
        running_sse_train = 0
        for user_idx in range(self.n_users):
            for item_idx in range(self.n_items):
                actual_rating = self.train_mat[user_idx, item_idx]

                if not np.isnan(actual_rating):
                    predicted_rating = self.predict(user_idx, item_idx)
                    error = actual_rating - predicted_rating
                    running_sse_train += error ** 2
                    self.update_biases(user_idx, item_idx, error)

                    # update the latent factors using stochastic gradient descent with regularization
                    for k in range(self.cfg_train.getint("latent_factors")):
                        self.user_factors[user_idx, k] += self.lrs[epoch] * (
                                error * self.item_factors[k, item_idx]
                                - self.cfg_train.getfloat("l2_reg") * self.user_factors[user_idx, k]
                        )
                        self.item_factors[k, item_idx] += self.lrs[epoch] * (
                                error * self.user_factors[user_idx, k]
                                - self.cfg_train.getfloat("l2_reg") * self.item_factors[k, item_idx]
                        )
        return running_sse_train / self.n_train_ratings


    def evaluate(self):
        """
        Evaluate the model on the validation set
        :return: mse val loss
        """
        running_sse_val = 0

        for user_idx in range(self.n_users):
            for item_idx in range(self.n_items):
                actual_rating = self.val_mat[user_idx, item_idx]
                if not np.isnan(actual_rating):
                    predicted_rating = self.predict(user_idx, item_idx)
                    error = actual_rating - predicted_rating
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
        print(f"Epoch: {epoch + 1}/{self.cfg_train.getint('num_epochs')} | Train MSE: {mse_train} | Val MSE: {mse_val}")


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


    def update_biases(self, user_idx, item_idx, error):
        self.user_bias[user_idx] += self.cfg_train.getfloat("bias_lr") \
                                    * (error - self.cfg_train.getfloat("bias_reg") * self.user_bias[user_idx])
        self.item_bias[item_idx] += self.cfg_train.getfloat("bias_lr") \
                                    * (error - self.cfg_train.getfloat("bias_reg") * self.item_bias[item_idx])


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

        for user_idx in range(matrix.shape[0]):
            user_row = matrix[user_idx, :]
            rated_indices = np.where(user_row != 0)[0] if unseen_mode == "zero" else np.where(~np.isnan(user_row))[0]
            n_rated = len(rated_indices)
            train_size = int(n_rated * train_ratio)
            train_indices = np.random.choice(rated_indices, size=train_size, replace=False)
            test_indices = np.setdiff1d(rated_indices, train_indices)
            train_matrix[user_idx, train_indices] = user_row[train_indices]
            test_matrix[user_idx, test_indices] = user_row[test_indices]

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
