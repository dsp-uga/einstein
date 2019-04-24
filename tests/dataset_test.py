import findspark
findspark.init()

import os
from einstein.dataset import Loader
from pyspark.sql import SparkSession


l = Loader()
spark = SparkSession.builder.master("yarn").appName(
	"Solar Irradiance Prediction").getOrCreate()


def test_load_data():
	df = l.load_data()
	# Testing that there is atleast one row in the Spark DataFrame
	assert df.count() > 0
	# Testing that there is atleast one column in the Spark DataFrame
	assert len(df.columns) > 0


def test_process_data():
	df = spark.read.csv(os.path.join(l.bucket, l.filename),
		header='true', inferSchema='true')
	processed_df = l.process_data(df)
	cols = processed_df.columns
	# Testing if 'label' column is present in the processed DataFrame
	assert 'label' in cols
	
	y_cols = [('y' + str(offset)) for offset in range(1, 
		25) if offset != l.target_hour]
	# Testing if other target hour offset columns are present in the processed
	# Spark DataFrame
	for y_col in y_cols:
		assert y_col not in cols


def get_input_columns():
	df = spark.read.csv(os.path.join(l.bucket, l.filename),
		header='true', inferSchema='true')
	processed_df = l.process_data(df)
	assert 'label' not in l.get_input_columns(processed_df)
