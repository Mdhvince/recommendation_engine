import configparser
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.mf_dataset import MFDataset
from src.ncf import NCFModel


if __name__ == "__main__":
    config = configparser.ConfigParser(inline_comment_prefixes="#")
    config.read(Path(__file__).parent.parent / "config.ini")

    training_data = MFDataset(config, mode="train")
    validation_data = MFDataset(config, mode="val")
    train_dataloader = DataLoader(training_data, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(validation_data, batch_size=32, shuffle=True)

    # get the number of users and items
    cfd_data = config["DATA"]
    data_dir = Path(__file__).parent.parent / Path(cfd_data.get("data_dirname"))
    ratings_path = data_dir / Path(cfd_data.get("ratings_filename"))
    ratings_mat = MFDataset.load_ratings_matrix(ratings_path)
    n_users = ratings_mat.shape[0]
    n_items = ratings_mat.shape[1]

    model = NCFModel(config, n_users, n_items)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(10):
        running_loss_train = 0.0
        running_val_loss = 0.0

        model.train()
        for batch in train_dataloader:
            user_idx = batch["user_idx"].to(device)
            item_idx = batch["item_idx"].to(device)
            rating = batch["rating"].to(device)
            prediction = model(user_idx, item_idx)
            loss = criterion(prediction, rating.unsqueeze(1))
            running_loss_train += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()

        for batch in val_dataloader:
            user_idx = batch["user_idx"].to(device)
            item_idx = batch["item_idx"].to(device)
            rating = batch["rating"].to(device)
            prediction = model(user_idx, item_idx)
            loss = criterion(prediction, rating.unsqueeze(1))
            running_val_loss += loss.item()

        train_loss = running_loss_train / len(train_dataloader)
        val_loss = running_val_loss / len(val_dataloader)
        print(f"Epoch {epoch + 1} | Train loss: {train_loss:.3f} | Val loss: {val_loss:.3f}")