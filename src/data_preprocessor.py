import configparser
import json
import pickle
import re
from pathlib import Path

import pyspark.sql.functions as F
from pyspark.ml.feature import StringIndexer
from pyspark.sql import Window
from pyspark.sql.types import StringType

from src.spark_dataframe import SparkDataframe


class DataPreprocessor:
    def __init__(self, config_db):
        self.spark_dataframe = SparkDataframe(config_db)
        self.steam_games = self.spark_dataframe("steam_games")
        self.max_rating = 10
        self.min_games = 5
        self.split_percentage = 0.2

    def preprocess(self):
        """
        """
        play_records = F.col("behavior") == F.lit("play")
        played = F.col("hours") > F.lit(0)
        window_user = Window.partitionBy("user_id")
        ratings_df = (
            self.steam_games
            .filter(play_records & played)
            .groupby("user_id", "game").agg(F.sum("hours").alias("hours"))
            .withColumn("max_hours_of_user", F.max("hours").over(window_user))
            .withColumn("rating", (F.col("hours") / F.col("max_hours_of_user")) * self.max_rating)
            .withColumn("n_games", F.count("game").over(window_user))  # count and not distinct because of the groupby
            .filter(F.col("n_games") >= self.min_games)
        )
        user_indexer = StringIndexer(inputCol="user_id", outputCol="user_index")
        game_indexer = StringIndexer(inputCol="game", outputCol="game_index")
        ratings_df = user_indexer.fit(ratings_df).transform(ratings_df)
        ratings_df = game_indexer.fit(ratings_df).transform(ratings_df)

        # prepare the data for splitting
        window_spec = Window.partitionBy("user_index")
        # shuffle the rows of each user
        ratings_df = ratings_df.withColumn("rand", F.rand())
        ratings_df = ratings_df.withColumn("row_num", F.row_number().over(window_spec.orderBy("rand")))

        ratings_df = ratings_df.withColumn("max_row_num", F.max("row_num").over(window_spec))

        ratings_df = self.split_spark_dataframe(ratings_df)
        return ratings_df

    def split_spark_dataframe(self, ratings_df):
        ratings_df = (
            ratings_df
            .withColumn("n_val", F.ceil(self.split_percentage * F.col("max_row_num")))
            .withColumn("n_train", F.col("max_row_num") - F.col("n_val"))
        )
        ratings_df = ratings_df.withColumn(
            "dataset", F.when(F.col("row_num") <= F.col("n_train"), "train").otherwise("val")
        )
        user_train = ratings_df.filter(F.col("dataset") == "train").select("user_index").distinct().count()
        user_val = ratings_df.filter(F.col("dataset") == "val").select("user_index").distinct().count()
        assert user_train == user_val, "Number of unique users in train and val should be the same"

        return ratings_df

    @staticmethod
    def save_info(df):
        df = df.select(
            F.col("user_index").cast("int"),
            F.col("game_index").cast("int"),
            "user_id",
            "game",
            "dataset",
            "rating"
        )
        msg = "The max index of users should be equal to the number of users"
        assert df.select(F.max("user_index")).collect()[0][0] == df.select("user_index").distinct().count() - 1, msg
        msg = "The max index of items should be equal to the number of items"
        assert df.select(F.max("game_index")).collect()[0][0] == df.select("game_index").distinct().count() - 1, msg
        assert df.columns[-1] == "rating", "The last column should be the rating"

        result_dict = dict(df.rdd.map(lambda row: (f"{row[:-1]}", row["rating"])).collect())

        data_dir = Path(__file__).parent.parent / "data_inference"
        with open(data_dir / "interactions.pkl", "wb") as f:
            pickle.dump(result_dict, f)

        stats = {
            "n_users": df.select("user_index").distinct().count(),
            "n_items": df.select("game_index").distinct().count(),
            "mean_rating": df.select(F.mean("rating")).collect()[0][0],
            "n_train_ratings": df.filter(F.col("dataset") == "train").count(),
            "n_val_ratings": df.filter(F.col("dataset") == "val").count()
        }

        with open(data_dir / "stats.json", "w") as f:
            json.dump(stats, f)

    @staticmethod
    @F.udf(returnType=StringType())
    def clean_text(text: str):
        text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
        return text

    def terminate(self):
        self.spark_dataframe.spark.stop()


if __name__ == "__main__":
    config_db = configparser.ConfigParser(inline_comment_prefixes="#")
    config_db.read(Path(__file__).parent.parent / "config_db.ini")

    dp = DataPreprocessor(config_db)
    ratings = dp.preprocess()
    dp.save_info(ratings)

    dp.terminate()

    # to submit the script to spark cluster:
    # - locate where the spark-submit command is on the system `which spark-submit`. So if using AWS EMR type the cmd in
    #   the Machine terminal.
    # - run the command `/usr/bin/spark-submit --master yarn /path/to/pyspark_script.py`
