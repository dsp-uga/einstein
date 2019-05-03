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
from pyspark.ml.tuning import ParamGridBuilder


class IsotonicRegressor(Model):
    """Subclasses base :class: `Model` to initialize an Isotonic Regression
    """
    def __init__(self, input_cols, **kwargs):
        """Initialises the IsotonicRegressor class.

        Args:
            input_cols (list):
                List of all the input column names
            **kwargs (dict):
                Dictionary of keyword arguments for user defined parameters
        """
        self.input_cols = input_cols
        self.kwargs = {k: v for k, v in kwargs.items() if v is not None}
        self.metrics = ["r2", "mae", "rmse"]


    def get_parameters(self, user_params):
        """Declares a dictionary of hyperparameters for Regression.

        Args:
            user_params (dict):
                Dictionary of keyword arguments for user defined parameters
        Returns:
            (dict):
                Dictionary containing all the parameters
        """
        parameter_dict = {"featuresCol": "scaledFeatures",
                          "isotonic": True}
        parameter_dict.update(user_params)
        for k, v in parameter_dict.items():
            print(f'{k} : {v}')
        return parameter_dict

    def model_define(self):
        """Returns a model with the hyperparameters inputted in :func:
        `get_parameters`

        Returns:
            (pyspark.ml.regression.DecisionTreeRegressor):
                Decision Tree Regression model
        """
        return IsotonicRegression()

    def get_param_grid(self, model):
        """Defines a parameter grid for crossvalidation with the user-defined
        model parameters
    
        Args:
            model (pyspark.ml.regression):
                Regression model for which ParamGrid is to be built
        Returns:
            param_grid (ParamGrid):
                Grid of tunable hyperparameters
        """
        params = self.get_parameters(self.kwargs)
        param_grid = ParamGridBuilder()\
            .addGrid(model.isotonic, [params['isotonic']])\
            .build()
        return param_grid
