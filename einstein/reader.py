"""
This script implements the :method: `read_csv`, which decrypts the data,
and deletes the data once read.

Author:
----------
Anirudh K.M. Kakarlapudi
"""
import os
import findspark
from einstein.utils.security import Decryption
findspark.init()
from pyspark.sql import SparkSession


def read_csv(file_name, encrypt_path='gs://profinal/',
             decrypt_path='gs://dsp_uga/'):
    """Reads the spark dataframe

    Args:
        file_name(str):
            Name of the csv file to be read as spark dataframe
        decrypt_path(str):
            A google storage path where the decrypt files should be stored
        encrypt_path(str):
            A google storage path where the enrypted files are stored
    Returns:
        data_frame:(data frame)
    """
    spark = SparkSession.builder.master("yarn").appName(
        "Solar Irradiance Prediction").getOrCreate()
    dec = Decryption(encrypt_path, decrypt_path)
    dec.decrypt(file_name)
    data_frame = spark.read.csv(os.path.join(decrypt_path, file_name),
                                header='true', inferSchema='true')
    return data_frame
