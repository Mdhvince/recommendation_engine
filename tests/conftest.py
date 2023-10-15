import configparser
from pathlib import Path

import numpy as np
import pytest

from src.matrix_factorization import MatrixFactorization


@pytest.fixture
def mf():
    config = configparser.ConfigParser(inline_comment_prefixes="#")
    config.read(Path(__file__).parent.parent / "config.ini")
    return MatrixFactorization(config)


@pytest.fixture
def rating_mat_nan():
    return np.array([
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        [np.nan, 63, 9, 0, 0, 2, 1, 3, 1, np.nan],
        [0, 0, 1, 1, 9, 18, 17, 86, 0, 0]
    ])


@pytest.fixture
def rating_mat_zero():
    return np.array([
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        [0, 63, 9, 0, 0, 2, 1, 3, 1, 0],
        [0, 0, 1, 1, 9, 18, 17, 86, 0, 0]
    ])
