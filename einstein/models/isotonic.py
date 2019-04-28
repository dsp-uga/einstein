"""
This script implements the :class: `IsotonicRegressor`, sub-classed on :class:
`Model`, which implements the Isotonic Regression model.

Author:
----------
Jayant Parashar
"""
import findspark
findspark.init()

from pyspark.ml.regression import IsotonicRegression
from einstein.models.base import Model


class IsotonicRegressor(Model):
    """Subclasses base :class: `Model` to initialize an Isotonic Regression
    """
    def __init__(self, input_cols, **kwargs):
        """Initialises the class.

        Args:
            input_cols(list):
                A list of all the input column names
            **kwargs:
                keyword arguments of user defined parameters
        """
        self.input_cols = input_cols
        self.kwargs = {k: v for k, v in kwargs.items() if v is not None}
        self.metrics = ["r2", "mae", "rmse"]

    def get_parameters(self, **user_params):
        """Declares a dictionary of hyperparameters for Regression.

        Args:
          **user_params:
            key word arguments of user defined parameters
        Returns:
            A  dictionary containing all the parameters
        """
        parameter_dict = {"featuresCol": "scaledFeatures",
                          "isotonic": True}
        parameter_dict.update(**user_params)
        for k, v in parameter_dict.items():
            print(f'{k} : {v}')
        return parameter_dict

    def model_define(self):
        """Returns a model with the hyperparameters inputted in :func:
        `get_parameters`.

        Returns:
            An Isotonic Regression model
        """
        params = self.get_parameters(**self.kwargs)
        return IsotonicRegression(**params)
