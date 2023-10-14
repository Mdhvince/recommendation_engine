import configparser
from pathlib import Path
from collections import defaultdict

import numpy as np



class MatrixFactorization:
    def __init__(self, config):
        """
        :param config:
        """
        self.cfg_train = config["TRAINING"]
        self.cfg_train.getint("latent_factors")

        self.user_bias = defaultdict(lambda: 0)
        self.item_bias = defaultdict(lambda: 0)

        self.lrs = self.learning_rate_decay()
        self.nic = 0  # no improvement counter
        self.best_loss = float("Inf")
        self.best_user_factors, self.best_item_factors = None, None

    def predict(self, u_factors, i_factors, u_idx, i_idx, items_mean):
        predicted = np.dot(u_factors[u_idx, :], i_factors[:, i_idx])
        bias = items_mean + self.user_bias[u_idx] + self.item_bias[i_idx]
        predicted += bias
        return predicted


    def init_learnable_factors(self, n_users, n_items):
        user_factors = np.random.rand(n_users, self.cfg_train.getint("latent_factors"))
        item_factors = np.random.rand(self.cfg_train.getint("latent_factors"), n_items)
        return user_factors, item_factors


    def train(self, epoch, n_users, n_items, ui_train_mat, num_ratings_train, user_factors, item_factors,
              all_items_mean):
        """
        Learn the latent factors and biases using SGD
        :param epoch: current epoch
        :param n_users: number of users
        :param n_items: number of items
        :param ui_train_mat: user-item matrix for training
        :param num_ratings_train: number of ratings in the training set
        :param user_factors: user by latent factors matrix
        :param item_factors: latent factors by item matrix
        :param all_items_mean: mean of all items (used for the biases)
        :return: updated user_factors, item_factors, mse train loss
        """
        running_sse_train = 0
        for user_idx in range(n_users):
            for item_idx in range(n_items):
                actual_rating = ui_train_mat[user_idx, item_idx]

                if not np.isnan(actual_rating):
                    predicted_rating = self.predict(user_factors, item_factors, user_idx, item_idx, all_items_mean)
                    error = actual_rating - predicted_rating
                    running_sse_train += error ** 2
                    self.update_biases(user_idx, item_idx, error)

                    # update the latent factors using stochastic gradient descent with regularization
                    for k in range(self.cfg_train.getint("latent_factors")):
                        user_factors[user_idx, k] += self.lrs[epoch] * (
                                error * item_factors[k, item_idx]
                                - self.cfg_train.getfloat("l2_reg") * user_factors[user_idx, k]
                        )
                        item_factors[k, item_idx] += self.lrs[epoch] * (
                                error * user_factors[user_idx, k]
                                - self.cfg_train.getfloat("l2_reg") * item_factors[k, item_idx]
                        )
        return user_factors, item_factors, running_sse_train / num_ratings_train


    def evaluate(self, n_users, n_items, ui_val_mat, num_ratings_val, user_factors, item_factors, all_items_mean):
        """
        Evaluate the model on the validation set
        :param n_users: number of users
        :param n_items: number of items
        :param ui_val_mat: user-item matrix for validation
        :param num_ratings_val: number of ratings in the validation set
        :param user_factors: user by latent factors matrix
        :param item_factors: latent factors by item matrix
        :param all_items_mean: mean of all items (used for the biases)
        :return: mse val loss
        """
        running_sse_val = 0

        for user_idx in range(n_users):
            for item_idx in range(n_items):
                actual_rating = ui_val_mat[user_idx, item_idx]
                if not np.isnan(actual_rating):
                    predicted_rating = self.predict(user_factors, item_factors, user_idx, item_idx, all_items_mean)
                    error = actual_rating - predicted_rating
                    running_sse_val += error ** 2

        return running_sse_val / num_ratings_val


    def learn(self, user_item_mat):
        assert isinstance(user_item_mat, np.ndarray), "user_item must be a numpy array"

        n_users, n_items = user_item_mat.shape
        ui_train_mat, ui_val_mat = split(user_item_mat, train_ratio=0.8, unseen_mode="nan")

        num_ratings = np.sum(~np.isnan(user_item_mat))
        num_ratings_train = np.sum(~np.isnan(ui_train_mat))
        num_ratings_val = np.sum(~np.isnan(ui_val_mat))
        all_items_mean = np.sum(~np.isnan(user_item_mat)) / num_ratings

        user_factors, item_factors = self.init_learnable_factors(n_users, n_items)
        self.best_loss = float("Inf")
        self.nic = 0

        # Learning process
        for epoch in range(self.cfg_train.getint("num_epochs")):
            user_factors, item_factors, mse_train_loss = self.train(
                epoch, n_users, n_items, ui_train_mat, num_ratings_train, user_factors, item_factors, all_items_mean
            )
            mse_val_loss = self.evaluate(
                n_users, n_items, ui_val_mat, num_ratings_val, user_factors, item_factors, all_items_mean
            )
            if self.can_stop(epoch, mse_val_loss, user_factors, item_factors):
                break

            print(f"Epoch: {epoch + 1}/{self.cfg_train.getint('num_epochs')} | Train MSE: {mse_train_loss} | Val MSE: {mse_val_loss}")


    def can_stop(self, epoch, mse_val_loss, user_factors, item_factors):
        """
        Check if the model can stop learning and update the best factors if needed
        :param epoch:
        :param mse_val_loss:
        :param user_factors:
        :param item_factors:
        :return: True if the model can stop learning, False otherwise
        """
        if mse_val_loss < self.best_loss and epoch > (self.cfg_train.getint("num_epochs") * 0.1):
            self.best_loss = mse_val_loss
            self.best_user_factors = user_factors
            self.best_item_factors = item_factors
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



def split(matrix, train_ratio, unseen_mode="zero", seed=42):
    assert isinstance(matrix, np.ndarray), "matrix must be a numpy array"
    assert unseen_mode in ["zero", "nan"], "unseen_mode must be 'zero' or 'nan', got {unseen_mode} instead"

    np.random.seed(seed)

    train_matrix = np.zeros_like(matrix) if unseen_mode == "zero" else np.nan * np.zeros_like(matrix)
    test_matrix = np.zeros_like(matrix) if unseen_mode == "zero" else np.nan * np.zeros_like(matrix)

    for user_idx in range(matrix.shape[0]):
        user_row = matrix[user_idx, :]
        rated = np.where(user_row != 0)[0] if unseen_mode == "zero" else np.where(~np.isnan(user_row))[0]
        n_rated = len(rated)
        train_size = int(n_rated * train_ratio)
        train_indices = np.random.choice(rated, size=train_size, replace=False)
        test_indices = np.setdiff1d(rated, train_indices)
        train_matrix[user_idx, train_indices] = user_row[train_indices]
        test_matrix[user_idx, test_indices] = user_row[test_indices]

    return train_matrix, test_matrix


def load_numpy_matrix(path):
    assert isinstance(path, Path), "path must be a pathlib.Path object"
    assert path.exists(), f"path {path} does not exist"
    return np.load(path)


if __name__ == "__main__":
    config = configparser.ConfigParser(inline_comment_prefixes="#")
    config.read(Path(__file__).parent.parent / "config.ini")

    ui_mat = load_numpy_matrix(Path(__file__).parent.parent / "data" / "ui_mat.npy")
    mf = MatrixFactorization(config)
    mf.learn(ui_mat)
