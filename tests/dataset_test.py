"""
Unit Test - Testing units in `dataset.py`

Author:
-------
Aashish Yadavally
"""
import os
import findspark
findspark.init()

from einstein.dataset import Loader
from pyspark.sql import SparkSession


L = Loader()
SPARKSESSION = SparkSession.builder.master("yarn").appName(
    "Solar Irradiance Prediction").getOrCreate()


def test_load_data():
    data_frame = L.load_data()
    # Testing that there is atleast one row in the Spark DataFrame
    assert data_frame.count() > 0
    # Testing that there is atleast one column in the Spark DataFrame
    assert len(data_frame.columns) > 0


def test_process_data():
    data_frame = SPARKSESSION.read.csv(os.path.join(L.bucket, L.filename),
                                       header='true', inferSchema='true')
    processed_df = L.process_data(data_frame)
    cols = processed_df.columns
    # Testing if 'label' column is present in the processed DataFrame
    assert 'label' in cols

    y_cols = [('y' + str(offset)) for offset in range(1, 25)
              if offset != L.target_hour]
    # Testing if other target hour offset columns are present in the processed
    # Spark DataFrame
    for y_col in y_cols:
        assert y_col not in cols


def get_input_columns():
    data_frame = SPARKSESSION.read.csv(os.path.join(L.bucket, L.filename),
                                       header='true', inferSchema='true')
    processed_df = L.process_data(data_frame)
    # Testing if 'label' is present in input columns list
    assert 'label' not in L.get_input_columns(processed_df)
