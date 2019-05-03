"""
This script implements the :class: `LinearRegressor`,  :class: `RidgeRegressor`
and :class: `LassoRegressor`, each sub-classed on :class: `Model`, which implement
the linear regression models.

Author:
----------
Jayant Parashar
"""
import findspark
findspark.init()

from pyspark.ml.tuning import ParamGridBuilder
from pyspark.ml.regression import LinearRegression
from einstein.models.base import Model


class LinearRegressor(Model):
    """A class for multiple linear regression extending an abstract class model.
    """
    def __init__(self, input_cols, **kwargs):
        """Initialises the LinearRegressor class.

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
        parameter_dict = {'maxIter': 20, 'regParam': 0.5,
                          'elasticNetParam': 0.5, 'tol': 1e-06,
                          'loss': 'squaredError', 'epsilon': 1.35}
        parameter_dict.update(user_params)
        for k, v in parameter_dict.items():
            print(f'{k} : {v}')
        return parameter_dict

    def model_define(self):
        """Returns a model with the hyperparameters inputted in :func:
        `get_parameters`

        Returns:
            (pyspark.ml.regression.LinearRegression):
                Linear Regression model
        """
        return LinearRegression()

    def get_param_grid(self, model):
        """Defines a parameter grid for crossvalidation with the user-defined
        model parameters
    
        Args:
            model (pyspark.ml.regression):
                Regression model
        """
        params = self.get_parameters(self.kwargs)
        param_grid = ParamGridBuilder()\
            .addGrid(model.maxIter, [params['maxIter']])\
            .addGrid(model.regParam, [params['regParam']])\
            .addGrid(model.elasticNetParam, [params['elasticNetParam']])\
            .addGrid(model.tol, [params['tol']])\
            .addGrid(model.loss, [params['loss']])\
            .addGrid(model.epsilon, [params['epsilon']])\
            .build()
        return param_grid


class RidgeRegressor(LinearRegressor):
    """A class that inherits from LinearRegressor class and implements
    Ridge regression
    """
    def get_parameters(self, user_params):
        """Defines a Ridge Regression model using the L2 Norm, on a dictionary
        of parameters

        Args:
            user_params (dict):
                Dictionary of keyword arguments for user defined parameters
        Returns:
            (dict):
                Dictionary containing all the parameters
        """
        parameter_dict = {'maxIter': 20, 'regParam': 0.5,
                          'elasticNetParam': 0.0, 'tol': 1e-06,
                          'loss': 'squaredError', 'epsilon': 1.35}
        parameter_dict.update(user_params)
        for k, v in parameter_dict.items():
            print(f'{k} : {v}')
        return parameter_dict

    def get_param_grid(self, model):
        """Defines a parameter grid for crossvalidation with the user-defined
        model parameters
    
        Args:
            model (pyspark.ml.regression):
                Regression model
        """
        params = self.get_parameters(self.kwargs)
        param_grid = ParamGridBuilder()\
            .addGrid(model.maxIter, [params['maxIter']])\
            .addGrid(model.regParam, [params['regParam']])\
            .addGrid(model.elasticNetParam, [params['elasticNetParam']])\
            .addGrid(model.tol, [params['tol']])\
            .addGrid(model.loss, [params['loss']])\
            .build()
        return param_grid


class LassoRegressor(LinearRegressor):
    """A class that inherits from LinearRegressor class and implements
    Lasso regression
    """
    def get_parameters(self, user_params):
        """Defines a Lasso Regression model using the L1 Norm, on a dictionary
        of parameters

        Args:
            user_params (dict):
                Dictionary of keyword arguments for user defined parameters
        Returns:
            (dict):
                Dictionary containing all the parameters
        """
        parameter_dict = {'maxIter': 20, 'regParam': 0.5,
                          'elasticNetParam': 1.0, 'tol': 1e-06,
                          'loss': 'squaredError', 'epsilon': 1.35}
        parameter_dict.update(user_params)
        for k, v in parameter_dict.items():
            print(f'{k} : {v}')
        return parameter_dict

    def get_param_grid(self, model):
        """Defines a parameter grid for crossvalidation with the user-defined
        model parameters
    
        Args:
            model (pyspark.ml.regression):
                Regression model
        """
        params = self.get_parameters(self.kwargs)
        param_grid = ParamGridBuilder()\
            .addGrid(model.maxIter, [params['maxIter']])\
            .addGrid(model.regParam, [params['regParam']])\
            .addGrid(model.elasticNetParam, [params['elasticNetParam']])\
            .addGrid(model.tol, [params['tol']])\
            .addGrid(model.loss, [params['loss']])\
            .build()
        return param_grid
