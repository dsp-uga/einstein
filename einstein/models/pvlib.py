"""
This script helps get the irradiance data for various PVLib Forecast Models,
and helps generate plots for their irradiance values.

Author:
-------
Aashish Yadavally
"""
import pandas as pd
import matplotlib.pyplot as plt
from einstein.models.base import Model
from pvlib.forecast import NAM, GFS, HRRR, RAP, NDFD


ATHENS_LAT, ATHENS_LON = 33.9052058, -83.382608

class PVLibModel(Model):
    """A module using :mod: `pvlib` for generating 'GHI', 'DHI' and 'DNI'
    irradiance values.
    """
    def __init__(self, model_name='NAM', lat=ATHENS_LAT, lon=ATHENS_LON,
                 start=None, stop=None):
        """Constructs a :mod: `pvlib` :class: `ForecastModel`

        Args:
            model_name (string):
                Name of PVLib model
            lat (float):
                The latitude value
            lon (float):
                The longitude value
            start (pd.Timestamp):
                The start time
            stop (pd.Timestamp):
                The stop time
        """
        self.lat = lat or ATHENS_LAT
        self.lon = lon or ATHENS_LON
        # If no `start` reftime is given, it is defaulted to current timestamp
        if start is None:
            self.start = pd.Timestamp.now()
            if stop is None:
                # If no `stop` reftime is given, it is defaulted to one day
                # from the `start` reftime
                self.stop = pd.Timestamp.now() + pd.Timedelta(1, 'D')
            elif self.start > stop:
                raise ValueError("'start' reftime should be less than 'stop'\
                                reftime.")
        elif stop is None:
            # If no `stop` reftime is given, it is defaulted to one day from
            # the `start` reftime
            self.start = start
            self.stop = self.start + pd.Timedelta(1, 'D')
        self.model_name = model_name or 'NAM'
        self.forecast_model = self.model_define()

    def model_define(self):
        """Defines the PVLib Model which is to be instantiated.

        Returns:
            :mod: `pvlib` :class: `ForecastModel`
        """
        if self.model_name == 'GFS':
            return GFS()
        elif self.model_name == 'HRRR':
            return HRRR()
        elif self.model_name == 'RAP':
            return RAP()
        elif self.model_name == 'NDFD':
            return NDFD()
        else:
            return NAM()

    def get_parameters(self):
        """A method that defines parameters for the model

        NOTE: PVLib Models do not require parameters like the Spark MLLib
        regression models, thus, this method does not return anything
        """
        return

    def get_processed_data(self):
        """Retrieves the data from the forecast model and returns features for
        each model

        Returns:
            data_frame (pd.DataFrame):
                Pandas Dataframe consisting of forecast model data
        """
        data_frame = self.forecast_model.get_processed_data(
            self.lat, self.lon, self.start, self.stop)
        return data_frame

    def get_irradiance(self, data_frame):
        """Returns the 'GHI', 'DHI' and 'DNI' irradiance values

        Args:
            data_frame (pd.DataFrame):
                Pandas DataFrame consisting of forecast model data

        Returns:
            irrads (list):
                List containing 'GHI', 'DHI' and 'DNI' values for the given
                time range, at the given location
        """
        irrads = self.forecast_model.cloud_cover_to_irradiance(
            data_frame['total_clouds'])
        return irrads

    def plot(self):
        """Plots the 'GHI', 'DHI' and 'DNI' irradiance values
        """
        data_frame = self.get_processed_data()
        irrads = self.get_irradiance(data_frame)
        irrads.plot()
        plt.ylabel('Irradiance ($W/m^2$)')
        plt.xlabel('Forecast Time')
        plt.title(f'{self.model_name} forecast for lat={self.lat}, lon={self.lon}')
        plt.legend()
