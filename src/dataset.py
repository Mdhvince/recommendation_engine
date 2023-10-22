import pickle
import re
from pathlib import Path

import pyspark
import pyspark.sql.functions as F
from pyspark.ml.feature import StringIndexer
from pyspark.sql import SparkSession, Window
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.types import StringType


class SparkDataframe:
    def __init__(self):
        self.driver = "org.sqlite.JDBC"
        driver_path = "/home/medhyvinceslas/.config/JetBrains/PyCharm2023.2/jdbc-drivers/Xerial SQLiteJDBC/3.43.0/org/xerial/sqlite-jdbc/3.43.0.0/sqlite-jdbc-3.43.0.0.jar"
        self.jdbc_url = "jdbc:sqlite:/home/medhyvinceslas/Documents/programming/recommendation_engine/identifier.sqlite"

        self.spark = (
            SparkSession.builder
            .appName("SteamDataPrep")
            .config("spark.jars", driver_path)
            .config("spark.driver.extraClassPath", driver_path)
            .getOrCreate()  # if already exists, get it
        )


    def __call__(self, dbtable: str) -> DataFrame:
        df = (
            self.spark.read.format("jdbc")
            .options(url=self.jdbc_url, dbtable=dbtable)
            .options(driver=self.driver)
            .load()
        )
        # df = self.spark.sparkContext.parallelize(df.collect()).toDF()
        return df


class DataProcessor:
    def __init__(self):
        self.spark_dataframe = SparkDataframe()
        self.steam_games = self.spark_dataframe("steam_games")
        self.max_rating = 10
        self.min_games = 5
        self.split_percentage = 0.2

    def preprocess(self):
        """
        """
        play_records = F.col("behavior") == F.lit("play")
        played = F.col("hours") > F.lit(0)
        window_maxh_per_user = Window.partitionBy("user_id")
        ratings_df = (
            self.steam_games
            .filter(play_records & played)
            .groupby("user_id", "game").agg(F.sum("hours").alias("hours"))
            .withColumn("max_hours_of_user", F.max("hours").over(window_maxh_per_user))
            .withColumn("rating", (F.col("hours") / F.col("max_hours_of_user")) * self.max_rating)
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
        ratings_df = ratings_df.filter(F.col("max_row_num") > self.min_games)  # filter out users with less than 5 games

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
    def save_info(ratings_df):
        ratings_df = ratings_df.select(
            F.col("user_index").cast("int"),
            F.col("game_index").cast("int"),
            "user_id",
            "game",
            "dataset",
            "rating"
        )
        assert ratings_df.columns[-1] == "rating", "The last column should be the rating"
        result_dict = dict(ratings_df.rdd.map(lambda row: (f"{row[:-1]}", row["rating"])).collect())

        data_dir = Path(__file__).parent.parent / "data_inference"
        with open(data_dir / "interactions.pkl", "wb") as f:
            pickle.dump(result_dict, f)



    @staticmethod
    @F.udf(returnType=StringType())
    def clean_text(text: str):
        text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
        return text


    def terminate(self):
        self.spark_dataframe.spark.stop()


if __name__ == "__main__":
    dp = DataProcessor()
    ratings = dp.preprocess()
    dp.save_info(ratings)

    dp.terminate()

    # to submit the script to spark cluster:
    # - locate where the spark-submit command is on the system `which spark-submit`. So if using AWS EMR type the cmd in
    #   the Machine terminal.
    # - run the command `/usr/bin/spark-submit --master yarn /path/to/pyspark_script.py`
