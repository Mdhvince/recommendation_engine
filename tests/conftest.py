import pytest
from pyspark.sql import SparkSession


@pytest.fixture(scope="session")
def spark_session():
    spark = (
        SparkSession.builder
        .master("local[*]")
        .appName("pytest-pyspark-local-testing")
        .getOrCreate()
    )
    yield spark
    spark.stop()


@pytest.fixture(scope="session")
def ratings_df(spark_session):
    ratings_df = spark_session.createDataFrame([
            ("1", "game1", 10.0, 0.0, 0.0),
            ("1", "game2", 5.0, 0.0, 1.0),
            ("1", "game3", 2.5, 0.0, 2.0),
            ("2", "game1", 7.5, 1.0, 0.0),
            ("2", "game2", 2.5, 1.0, 1.0),
            ("2", "game3", 10.0, 1.0, 2.0),
            ("3", "game1", 2.5, 2.0, 0.0),
            ("3", "game2", 10.0, 2.0, 1.0),
            ("3", "game3", 7.5, 2.0, 2.0),
            ("4", "game1", 2.5, 3.0, 0.0),
            ("4", "game2", 10.0, 3.0, 1.0),
            ("4", "game3", 7.5, 3.0, 2.0),
            ("5", "game1", 2.5, 4.0, 0.0),
            ("5", "game2", 10.0, 4.0, 1.0),
            ("5", "game3", 7.5, 4.0, 2.0),
            ("6", "game1", 2.5, 5.0, 0.0),
            ("6", "game2", 10.0, 5.0, 1.0),
            ("6", "game3", 7.5, 5.0, 2.0),
            ("7", "game1", 2.5, 6.0, 0.0),
            ("7", "game2", 10.0, 6.0, 1.0),
            ("7", "game3", 7.5, 6.0, 2.0),
            ("8", "game1", 2.5, 7.0, 0.0),
            ("8", "game2", 10.0, 7.0, 1.0),
            ("8", "game3", 7.5, 7.0, 2.0)],
        ["user_id", "game", "rating", "user_index", "game_index"]
    )
    return ratings_df
