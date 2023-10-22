## Matrix Factorization SVD++ for Recommendation System

### Preprocessing with PySpark
To submit the script to spark cluster:
- ssh into the cluster
- locate where the spark-submit command is on the system `which spark-submit`. So if using AWS EMR type the cmd in the Machine terminal.
- upload the dataset.py to the cluster with `scp`
- run the command `/usr/bin/spark-submit --master yarn /path/to/dataset.py` or [follow the instructions if using Pycharm](https://www.jetbrains.com/help/pycharm/big-data-tools-spark-submit.html)

