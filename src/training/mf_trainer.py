import configparser
import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from src.utils.user_item_indexer import UserItemIndexer


class MFTrainer:
    def __init__(self, config):

        logging.basicConfig(filename="mf_trainer.log", level=logging.INFO, format="%(asctime)s %(message)s")
        self.logger = logging.getLogger(__name__)

        self.cfg_data = config["DATA"]
        self.cfg_train = config["TRAINING"]

        self.N_FACTORS = self.cfg_train.getint("latent_factors")
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.L2 = self.cfg_train.getfloat("l2_reg")
        self.lrs = self.learning_rate_decay()

        # Data for training and validation
        ui_indexer = UserItemIndexer(config)
        self.interactions_train, self.interactions_val = ui_indexer.split()

        stats = self.load_stats()
        self.N_USERS = stats["n_users"]
        self.N_ITEMS = stats["n_items"]
        self.RATING_AVG = stats["mean_rating"]
        self.N_TRAIN = stats["n_train_ratings"]
        self.N_VAL = stats["n_val_ratings"]
        self.user_factors, self.item_factors = self.init_learnable_factors()

        self.nic = 0  # no improvement counter
        self.best_loss = float("Inf")
        self.user_bias = defaultdict(lambda: 0)
        self.item_bias = defaultdict(lambda: 0)
        self.U, self.V_t = None, None

    def init_learnable_factors(self) -> (torch.FloatTensor, torch.FloatTensor):
        user_factors = torch.randn(self.N_USERS, self.N_FACTORS, device=self.DEVICE)
        item_factors = torch.randn(self.N_FACTORS, self.N_ITEMS, device=self.DEVICE)
        return user_factors, item_factors

    def load_stats(self):
        stats_path = Path(__file__).parent.parent / "data_inference" / self.cfg_data.get("stats_filename")
        with open(stats_path, "rb") as f:
            stats = json.load(f)
        return stats


    def results_generator(self, interaction_dict):
        """
        Generator to iterate over user-item interactions
        """
        for key, actual_rating in interaction_dict.items():
            user_index, item_index = eval(key)
            predicted_rating: torch.FloatTensor = self.predict(user_index, item_index)
            error: torch.FloatTensor = actual_rating - predicted_rating
            yield user_index, item_index, actual_rating, predicted_rating, error


    def predict(self, u_idx: int, i_idx: int) -> torch.FloatTensor:
        predicted = torch.dot(self.user_factors[u_idx, :], self.item_factors[:, i_idx])
        bias = self.RATING_AVG + self.user_bias[u_idx] + self.item_bias[i_idx]
        predicted += bias
        return predicted


    def optimize(self, epoch: int, user_index: int, item_index: int, error: float):
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

        for k in range(self.N_FACTORS):
            user_factor_gradient = error * self.item_factors[k, item_index] - self.L2 * self.user_factors[user_index, k]
            item_factor_gradient = error * self.user_factors[user_index, k] - self.L2 * self.item_factors[k, item_index]

            user_factors_momentum = momentum * user_factors_momentum + self.lrs[epoch] * user_factor_gradient
            item_factors_momentum = momentum * item_factors_momentum + self.lrs[epoch] * item_factor_gradient

            self.user_factors[user_index, k] += user_factors_momentum
            self.item_factors[k, item_index] += item_factors_momentum


    def train(self, epoch: int):
        """
        Learn the latent factors and biases using SGD
        :param epoch: current epoch
        :return: updated user_factors, item_factors, mse train loss
        """
        running_sse_train = 0

        for user_index, item_index, actual_rating, pred, error in self.results_generator(self.interactions_train):
            running_sse_train += error ** 2
            self.update_biases(user_index, item_index, error)
            self.optimize(epoch, user_index, item_index, error)

        return running_sse_train / self.N_TRAIN


    def evaluate(self):
        """
        Evaluate the model on the validation set
        :return: mse val loss
        """
        running_sse_val = 0
        for *_, error in self.results_generator(self.interactions_val):
            running_sse_val += error ** 2

        return running_sse_val / self.N_VAL


    def learn(self):

        for epoch in range(self.cfg_train.getint("num_epochs")):
            mse_train = self.train(epoch)
            mse_val = self.evaluate()
            if self.can_stop(epoch, mse_val): break
            self.log(epoch, mse_train, mse_val)


    def log(self, epoch, mse_train, mse_val):
        self.logger.info(
            f"Epoch: {epoch + 1}/{self.cfg_train.getint('num_epochs')} "
            f"| Train RMSE - {np.sqrt(mse_train):.3f} | Val RMSE - {np.sqrt(mse_val):.3f}"
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
            self.U = self.user_factors
            self.V_t = self.item_factors
            self.nic = 0  # no improvement counter
        else:
            if epoch > (self.cfg_train.getint("num_epochs") * 0.1):
                self.nic += 1
        
        if self.cfg_train.getboolean("early_stopping") and (self.nic >= self.cfg_train.getint("patience")):
            return True
        return False


    def update_biases(self, user_index: int, item_index: int, error: torch.FloatTensor):
        self.user_bias[user_index] += self.cfg_train.getfloat("bias_lr") \
                                    * (error.item() - self.cfg_train.getfloat("bias_reg") * self.user_bias[user_index])
        self.item_bias[item_index] += self.cfg_train.getfloat("bias_lr") \
                                    * (error.item() - self.cfg_train.getfloat("bias_reg") * self.item_bias[item_index])


    def learning_rate_decay(self):
        learning_rates = []
        for epoch in range(self.cfg_train.getint("num_epochs")):
            lr = self.cfg_train.getfloat("end_lr") \
                 + (self.cfg_train.getfloat("start_lr") - self.cfg_train.getfloat("end_lr")) \
                 * np.exp(-self.cfg_train.getfloat("decay_rate") * epoch)
            learning_rates.append(lr)
        return learning_rates


    def save(self):
        root_dir = Path(__file__).parent.parent
        data_inference_dir = root_dir / "data_inference"
        data_inference_dir.mkdir(parents=True, exist_ok=True)
        user_factors_path = data_inference_dir / self.cfg_data.get("user_factors_filename")
        item_factors_path = data_inference_dir / self.cfg_data.get("item_factors_filename")
        user_bias_path = data_inference_dir / self.cfg_data.get("user_bias_filename")
        item_bias_path = data_inference_dir / self.cfg_data.get("item_bias_filename")

        torch.save(self.U, user_factors_path)
        torch.save(self.V_t, item_factors_path)

        with open(user_bias_path, "w") as f:
            json.dump(dict(self.user_bias), f)
        with open(item_bias_path, "w") as f:
            json.dump(dict(self.item_bias), f)

        print(f"Saved user and item factors to {user_factors_path} and {item_factors_path}")
        print(f"Saved user and item biases to {user_bias_path} and {item_bias_path}")



if __name__ == "__main__":
    config = configparser.ConfigParser(inline_comment_prefixes="#")
    config.read(Path(__file__).parent.parent / "config.ini")

    mf = MFTrainer(config)
    mf.learn()
    # mf.save()


