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


def test_get_interactions_map(ratings_df):
    # Arrange
    split_percentage = 0.2
    df = DataPreprocessor.associate_splits(ratings_df, split_percentage)
    # user_index, game_index, user_id, game, dataset, rating

    # Act
    result_dict = DataPreprocessor.get_interactions_map(df)

    # Assert
    assert len(result_dict) == 24, "The number of interactions should be 24 (8 users x 3 games)"
    assert result_dict["(0, 0, '1', 'game1', 'train')"] == 10.0, "The rating should be 10.0"
    assert len([key for key in result_dict.keys() if eval(key)[4] == "train"]) == 16, "The number of train interactions should be 16 (train percentage = 0.8)"
    assert len([key for key in result_dict.keys() if eval(key)[4] == "val"]) == 8, "The number of val interactions should be 8 (val percentage = 0.2)"



