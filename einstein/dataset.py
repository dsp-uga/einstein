"""
This script defines :class: `Loader` and :class: `_Fetcher`. The `Loader`
class loads the CSV file stored in the Google Storage Bucket, and processes
it by dropping non-target columns. The `_Fetcher` class can not be sub-classed
by importing the module, and was used by the developers to fetch the data from
the 'Institute of Artificial Intelligence' local servers, to fetch the dataset.

Author:
-----------
Aashish Yadavally
"""
import os
import logging
import datetime
import numpy as np
import pandas as pd
from einstein.reader import read_csv


# First reftime to be retrieved while fetching the data
FIRST = pd.Timestamp(2017, 1, 1, 0)
# Last reftime to be retrieved while fetching the data
LAST = pd.Timestamp(2017, 12, 31, 18)
# Grid sizes around ATHENS to be retained
GEO_SHAPES = [(1, 1), (3, 3), (5, 5)]


def load_data(**kwargs):
    """Initializes a Pyspark SparkSession and loads the data into a Spark
    Dataframe

        Returns:
            Spark DataFrame:
                Dataset read into a Spark DataFrame
    """
    l = Loader(**kwargs)
    return l.load_data()


class Loader:
    """Used to load the CSV file in the Google Storage Bucket, based on the
    grid shape, set the target hour offset for predictions, drop the
    non-target hour offset and initialize the Spark Dataframe
    """
    def __init__(self, target_hour=1, bucket='gs://uga_dsp_sp19',
                 filename='2017_3.csv'):
        """Initializes the :class: `Loader` and sets up the Spark Dataframe
        which can be used for model training

        Arguments:
            target_hour (int):
                Prediction for the hour offset in the reftime dimension
            bucket (str):
                Link to Google Storage Bucket which contains the CSV files
                containing the Solar data
            filename (str):
                Filename of the CSV file in Google Storage Bucket which is to
                be read from
        """
        self.bucket = bucket
        self.filename = filename
        self.target_hour = target_hour

    def load_data(self):
        """Initializes a Pyspark SparkSession and loads the data into a Spark
        Dataframe

        Returns:
            df (Spark DataFrame):
                Dataset read into a Spark Dataframe
        """
        self.df = read_csv(os.path.join(self.bucket, self.filename),
                                 header='true', inferSchema='true')
        self.processed_df = self.process_data(self.df)
        self.input_cols = self.get_input_columns(self.processed_df)
        return self.processed_df

    def process_data(self, df):
        """Processes the input dataframe and sets it up for regression problem

        Arguments:
            df (Spark DataFrame):
                Spark dataframe to be processed
        Returns:
            df (Spark DataFrame):
                Spark dataframe after removing columns containing non-target
                offset hours, and renaming the target offset hour column
        """
        temp = [('y' + str(hr)) for hr in range(1, 25)]
        label = 'y' + str(self.target_hour)
        temp.remove(label)
        non_target_cols = temp
        df_cols = df.columns
        # Renaming target column to 'label'
        df = df.withColumnRenamed(label, 'label')
        # Dropping non-target columns
        for non_target_col in non_target_cols:
            df = df.drop(non_target_col)
        return df

    def get_input_columns(self, df):
        """Returns list of input columns

        Arguments:
            df (Spark DataFrame):
                Dataframe of dataset, whose input-output column names need
                to be retrieved

        Returns:
            list:
                List of input column names
        """
        columns = df.columns
        columns.remove('label')
        return columns


class _Fetcher:
    """Composes :mod: `apollo` :class: `SolarDataset` which unifies the
    NAM-NMM data with different target variables in GA Power datasets for
    target hours between 1 and 24

    # NOTE: Starting the class with "_" gives it a private behavior, wherein,
    it cannot be imported by importing the module
    """
    def __init__(self, first=FIRST, last=LAST):
        """Initializes `Fetcher` class which fetches the data from Institute
        of Artificial Intelligence's local store, where the dataset is cached

        # NOTE: Class access limited to developers

        Arguments:
            first (pd.Timestamp):
                First reftime to be retrieved
            last (pd.Timestamp):
                Last reftime to be retrieved
        """

        for geo_shape in GEO_SHAPES:
            x, y = self.tabular(first, last, geo_shape)
            x = np.asarray(x)
            y = np.asarray(y)
            # Joining 'x' and 'y'
            xy = np.concatenate((x, y), axis=1)
            cols = np.asarray([[col for col in range(1, y.shape[1])]])
            # Adding cols as column names into the 2D array
            cxy = np.concatenate((cols, xy), axis=0)
            name = str(first) + '_' + str(last) + '_' + str(geo_shape) + '.csv'
            np.savetxt(name, cxy, delimiter=',')

    def tabular(self, first, last, geo_shape):
        """Returns a tabular version of the dataset, wherein, all spatial and
        temporal dimensions are flattened and concatenated into a single
        vector per instance

        Arguments:
            first (pd.Timestamp):
                First reftime to be retrieved
            last (pd.Timestamp):
                Last reftime to be retrieved
            geo_shape (tuple):
                Grid size around the ATHENS location which is retained
        """
        from apollo.models.solar import SolarDataset

        sd = SolarDataset(start=first, stop=last, geo_shape=geo_shape,
                          standardize=False)
        x, y = sd.tabular()
        return x, y
