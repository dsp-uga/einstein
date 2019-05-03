"""
This script implements the :class: `DTRegressor`,  :class: `RFRegressor`
and :class: `GBTRegressor`, each sub-classed on :class: `Model`, which
implement the tree regression models.

Author:
----------
Anirudh Kumar Maurya Kakarlapudi
"""
import findspark
findspark.init()

from pyspark.ml.tuning import ParamGridBuilder
from pyspark.ml.regression import (RandomForestRegressor,
                                   DecisionTreeRegressor,
                                   GBTRegressor)
from einstein.models.base import Model


class DTRegressor(Model):
    """Subclasses base :class: `Model` to initialize a Decision Tree Regressor
    """
    def __init__(self, input_cols, **kwargs):
        """Initialises the DTRegressor class.

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
                          "maxDepth": 4,
                          "maxBins": 32}
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
        return DecisionTreeRegressor()

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
            .addGrid(model.maxDepth, [params['maxDepth']])\
            .addGrid(model.maxBins, [params['maxBins']])\
            .build()
        return param_grid


class RFRegressor(Model):
    """Subclasses base :class: `Model` to initialize a Random Forest Regressor
    """
    def __init__(self, input_cols, **kwargs):
        """Initialises the RFRegressor class.

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
                          "numTrees": 20,
                          "maxDepth": 4,
                          "maxBins": 32}
        parameter_dict.update(user_params)
        for k, v in parameter_dict.items():
            print(f'{k} : {v}')
        return parameter_dict

    def model_define(self):
        """Returns a model with the hyperparameters inputted in :func:
        `get_parameters`.

        Returns:
            (pyspark.ml.regression.RandomForestRegressor)
                Random Forest Regression model
        """
        return RandomForestRegressor()

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
            .addGrid(model.maxDepth, [params['maxDepth']])\
            .addGrid(model.numTrees, [params['numTrees']])\
            .addGrid(model.maxBins, [params['maxBins']])\
            .build()
        return param_grid


class GBTreeRegressor(Model):
    """Subclasses base :class: `Model` to initialize a Gradient Boost Tree
    Regressor.
    """
    def __init__(self, input_cols, **kwargs):
        """Initialises the GBTRegressor class.

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
                          "maxDepth": 4,
                          "maxIter": 10,
                          "maxBins": 32}
        parameter_dict.update(user_params)
        for k, v in parameter_dict.items():
            print(f'{k} : {v}')
        return parameter_dict

    def model_define(self):
        """Returns a model with the hyperparameters inputted in :func:
        `get_parameters`

        Returns:
            (pyspark.ml.regression.GBTRegressor)
                Gradient Boosting Tree Regression model
        """
        return GBTRegressor()

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
            .addGrid(model.maxDepth, [params['maxDepth']])\
            .addGrid(model.maxIter, [params['maxIter']])\
            .addGrid(model.maxBins, [params['maxBins']])\
            .build()
        return param_grid
