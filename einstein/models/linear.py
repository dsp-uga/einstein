"""
A script to implement regression models
"""
from pyspark.ml.regression import LinearRegression
from base import Model


class LinearRegressor(Model):
    """A class for multiple linear regression extending an abstract class model.
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
        self.kwargs = kwargs
        self.metrics = ["r2", "mae", "rmse"]

    def get_parameters(self, **user_params):
        """A method that defines a dictionary of parameters for regression model

        Args:
          **user_params:
            key word arguments of user defined parameters
        Returns:
            A  dictionary containing all the parameters
        """
        parameter_dict = {'maxIter': 20, 'regParam': 0.5,
                          'elasticNetParam': 0.5, 'tol': 1e-06,
                          'loss': 'squaredError', 'epsilon': 1.35}
        parameter_dict.update(**user_params)
        return parameter_dict

    def model_define(self):
        """A method which defines the linear regression model

        Returns:
            A linear regression model
        """
        params = self.get_parameters(**self.kwargs)
        lr = LinearRegression(**params)
        return lr


class RidgeRegressor(LinearRegressor):
    """A class that inherits from LinearRegressor class and implements
    Ridge regression
    """
    def get_parameters(self, **user_params):
        """A method that defines a dictionary of parameters for ridge
        regression model by using L2 norm

        Args:
          **user_params:
            key word arguments of user defined parameters
        Returns:
            A  dictionary containing all the parameters
        """
        parameter_dict = {'maxIter': 20, 'regParam': 0.5,
                          'elasticNetParam': 0.0, 'tol': 1e-06,
                          'loss': 'squaredError', 'epsilon': 1.35}
        parameter_dict.update(**user_params)
        return parameter_dict


class LassoRegressor(LinearRegressor):
    """A class that inherits from LinearRegressor class and implements
    Lasso regression
    """
    def get_parameters(self, **user_params):
        """A method that defines a dictionary of parameters for lasso
        regression model by using L1 norm

        Args:
          **user_params:
            key word arguments of user defined parameters
        Returns:
            A  dictionary containing all the parameters
        """
        parameter_dict = {'maxIter': 20, 'regParam': 0.5,
                          'elasticNetParam': 1.0, 'tol': 1e-06,
                          'loss': 'squaredError', 'epsilon': 1.35}
        parameter_dict.update(**user_params)
        return parameter_dict
