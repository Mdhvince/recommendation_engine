import pytest
import numpy as np

from src.dl.mf_dataset import MFDataset
from tests.conftest import rating_mat_nan, rating_mat_zero


@pytest.mark.parametrize("unseen_mode", ["nan", "zero"])
def test_split(rating_mat_nan, rating_mat_zero, unseen_mode):
    # Arrange
    train_ratio = 0.8
    rating_mat = rating_mat_nan if unseen_mode == "nan" else rating_mat_zero

    # Act
    train_mat, val_mat = MFDataset.split(rating_mat, train_ratio, unseen_mode)

    # Assert
    assert train_mat.shape == rating_mat.shape
    assert val_mat.shape == rating_mat.shape
    non_nan_idx_train = np.argwhere(~np.isnan(train_mat)) if unseen_mode == "nan" else np.argwhere(train_mat != 0)
    non_nan_idx_val = np.argwhere(~np.isnan(val_mat)) if unseen_mode == "nan" else np.argwhere(val_mat != 0)
    intersection = set(tuple(row) for row in non_nan_idx_train) & set(tuple(row) for row in non_nan_idx_val)
    assert len(intersection) == 0
