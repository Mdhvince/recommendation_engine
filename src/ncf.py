import torch


class NCFModel(torch.nn.Module):
    """
    Neural Collaborative Filtering model for matrix factorization.
    """
    def __init__(self, config, n_users, n_items):
        super().__init__()
        latent_factors = config["TRAINING"].getint("latent_factors")
        self.user_embed = torch.nn.Embedding(n_users, latent_factors)
        self.item_embed = torch.nn.Embedding(n_items, latent_factors)
        self.user_bias = torch.nn.Embedding(n_users, 1)
        self.item_bias = torch.nn.Embedding(n_items, 1)
        self.dropout = torch.nn.Dropout(0.2)
        self.relu = torch.nn.ReLU()
        self.fc0 = torch.nn.Linear(2 * latent_factors, 128)
        self.fc1 = torch.nn.Linear(128, 256)
        self.fc2 = torch.nn.Linear(256, 128)
        self.fc3 = torch.nn.Linear(128, 64)
        self.fc4 = torch.nn.Linear(64, 32)
        self.out = torch.nn.Linear(32, 1)

    def forward(self, user_idx, item_idx):
        user_embed = self.dropout(self.user_embed(user_idx))
        item_embed = self.dropout(self.item_embed(item_idx))
        user_bias = self.dropout(self.user_bias(user_idx))
        item_bias = self.dropout(self.item_bias(item_idx))
        x = torch.cat([user_embed, item_embed], dim=1)
        x = self.dropout(self.relu(self.fc0(x)))
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.dropout(self.relu(self.fc3(x)))
        x = self.dropout(self.relu(self.fc4(x)))
        x = self.out(x)
        return x + user_bias + item_bias