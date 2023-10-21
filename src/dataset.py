import re
from pathlib import Path

import pyspark
import pyspark.sql.functions as F
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
            .getOrCreate()  # if already exists, get it
        )

        print(f"Spark UI available at: {self.spark.sparkContext.uiWebUrl}")

    def __call__(self, dbtable: str) -> DataFrame:
        df = (
            self.spark.read.format("jdbc")
            .options(url=self.jdbc_url, dbtable=dbtable)
            .options(driver=self.driver)
            .load()
        )
        df = self.spark.sparkContext.parallelize(df.collect()).toDF()
        return df


class DataProcessor:
    def __init__(self):
        self.spark_dataframe = SparkDataframe()
        self.steam_games = self.spark_dataframe("steam_games")
        self.max_rating = 10

    def preprocess(self):
        play_records = F.col("behavior") == F.lit("play")
        played = F.col("hours") > F.lit(0)
        window_maxh_per_user = Window.partitionBy("user_id")
        df = (
            self.steam_games
            .filter(play_records & played)
            .groupby("user_id", "game").agg(F.sum("hours").alias("hours"))
            .withColumn("max_hours_of_user", F.max("hours").over(window_maxh_per_user))
            .withColumn("normalized_hours", (F.col("hours") / F.col("max_hours_of_user")) * self.max_rating)
            .select("user_id", "game", F.col("normalized_hours").alias("rating"))
        )
        df = df.withColumn("game_cleaned", DataProcessor.clean_text(F.col("game")))
        df.show()


    @staticmethod
    @F.udf(returnType=StringType())
    def clean_text(text: str):
        text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
        return text


    def terminate(self):
        self.spark_dataframe.spark.stop()


if __name__ == "__main__":
    dp = DataProcessor()
    dp.preprocess()
    dp.terminate()

    # to submit the script to spark cluster:
    # - locate where the spark-submit command is on the system `which spark-submit`. So if using AWS EMR type the cmd in
    #   the Machine terminal.
    # - run the command `/usr/bin/spark-submit --master yarn /path/to/pyspark_script.py`

