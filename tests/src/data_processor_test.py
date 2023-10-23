import pyspark.sql.functions as F

from src.data_preprocessor import DataPreprocessor
from tests.conftest import ratings_df


def test_associate_splits(ratings_df):
    # Arrange
    split_percentage = 0.2

    # Act
    df = DataPreprocessor.associate_splits(ratings_df, split_percentage)

    # Assert
    n_user_train = df.filter(F.col("dataset") == "train").select("user_index").distinct().count()
    n_user_val = df.filter(F.col("dataset") == "val").select("user_index").distinct().count()
    assert n_user_train == n_user_val, "The number of users in train and val should be the same"
    assert n_user_val == 8, "The number of users in val should be 8"
    msg = "The max index of users should be equal to the number of users"
    assert df.select(F.max("user_index")).collect()[0][0] == df.select("user_index").distinct().count() - 1, msg
    msg = "The max index of items should be equal to the number of items"
    assert df.select(F.max("game_index")).collect()[0][0] == df.select("game_index").distinct().count() - 1, msg
    assert df.columns[-1] == "rating", "The last column should be the rating"


