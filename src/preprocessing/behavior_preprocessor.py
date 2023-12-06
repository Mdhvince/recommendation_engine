import configparser
import json
import pickle
import re
from pathlib import Path
from typing import Dict

import pyspark.sql.functions as F
from pyspark.ml.feature import StringIndexer
from pyspark.sql import DataFrame
from pyspark.sql import Window
from pyspark.sql.types import StringType

from src.utils.spark_dataframe import SparkDataframe


class BehaviorPreprocessor:
    def __init__(self, config_db):
        self.spark_dataframe = SparkDataframe(config_db)
        self.steam_games = self.spark_dataframe("steam_games")
        self.max_rating = 10
        self.min_games = 5
        self.split_percentage = 0.2

    def preprocess(self):
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

        ratings_df = BehaviorPreprocessor.associate_splits(ratings_df, self.split_percentage)
        return ratings_df

    @staticmethod
    def associate_splits(df, split_percentage):
        window_spec = Window.partitionBy("user_index")
        df = (
            df
            .withColumn("rand", F.rand())
            .withColumn("row_num", F.row_number().over(window_spec.orderBy("rand")))
            .withColumn("max_row_num", F.max("row_num").over(window_spec))
            .withColumn("n_val", F.ceil(split_percentage * F.col("max_row_num")))
            .withColumn("n_train", F.col("max_row_num") - F.col("n_val"))
            .withColumn("dataset", F.when(F.col("row_num") <= F.col("n_train"), "train").otherwise("val"))
            .select(F.col("user_index").cast("int"), F.col("game_index").cast("int"), "user_id", "game", "dataset", "rating")
        )
        return df

    @staticmethod
    def get_interactions_map(df: DataFrame) -> Dict[str, float]:
        result_dict = dict(df.rdd.map(lambda row: (f"{row[:-1]}", row["rating"])).collect())
        return result_dict

    @staticmethod
    def get_stats(df: DataFrame):
        stats = {
            "n_users": df.select("user_index").distinct().count(),
            "n_items": df.select("game_index").distinct().count(),
            "mean_rating": df.select(F.mean("rating")).collect()[0][0],
            "n_train_ratings": df.filter(F.col("dataset") == "train").count(),
            "n_val_ratings": df.filter(F.col("dataset") == "val").count()
        }
        return stats

    @staticmethod
    def save_info(df: DataFrame, interactions: Dict[str, float], stats: Dict[str, float]):
        data_dir = Path(__file__).parent.parent / "data_inference"
        with open(data_dir / "interactions.pkl", "wb") as f: pickle.dump(interactions, f)
        with open(data_dir / "stats.json", "w") as f: json.dump(stats, f)

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

    dp = BehaviorPreprocessor(config_db)

    ratings = dp.preprocess()
    interactions_map = BehaviorPreprocessor.get_interactions_map(ratings)
    stats = BehaviorPreprocessor.get_stats(ratings)
    BehaviorPreprocessor.save_info(ratings, interactions_map, stats)

    dp.terminate()
