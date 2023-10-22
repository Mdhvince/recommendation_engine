from pyspark.sql import SparkSession, DataFrame


class SparkDataframe:
    def __init__(self, config_db):
        self.cfg = config_db["DEFAULT"]
        self.driver = self.cfg.get("driver")
        driver_path = self.cfg.get("driver_path")
        self.jdbc_url = self.cfg.get("jdbc_url")

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
        df = self.spark.sparkContext.parallelize(df.collect()).toDF()
        n_cpus_cores = self.spark.sparkContext.defaultParallelism
        df = df.repartition(n_cpus_cores)
        return df
