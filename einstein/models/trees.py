"""
A Script to implement Regression using Trees
"""


from pyspark.ml.regression import RandomForestRegressor,\
                                  DecisionTreeRegressor,\
                                  GBTRegressor
from pyspark.sql.session import SparkSession
from pyspark import SparkConf, SparkContext
from base import Model


class DecisionTree(Model):
    """Subclasses base :class: `Model` to initialize a Decision Tree Regressor
    """
    def __init__(self, input_cols, **kwargs):
        """Initialises the class.

        Args:
            input_cols(list):
                A list of all the input column names
            **kwargs:
                keyword arguments of user defined parameters
        """
        self.input_cols = input_col
        self.kwargs = kwargs
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
                          "maxDepth": 4,
                          "maxBins": 32}
        parameter_dict.update(**user_params)
        return parameter_dict

    def model_define(self):
        """Returns a model with the hyperparameters inputted in :func:
        `get_parameters`.

        Returns:
            A Decision Tree Regression model
        """
        params = self.get_parameters(**self.kwargs)
        return DecisionTreeRegressor(**params)


class RF(Model):
    """Subclasses base :class: `Model` to initialize a Random Forest Regressor
    """
    def __init__(self, input_cols, **kwargs):
        """Initialises the class.

        Args:
            input_cols(list):
                A list of all the input column names
            **kwargs:
                keyword arguments of user defined parameters
        """
        self.input_cols = input_col
        self.kwargs = kwargs
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
                          "numTrees": 20,
                          "maxDepth": 4,
                          "maxBins": 32}
        parameter_dict.update(**user_params)
        return parameter_dict

    def model_define(self):
        """Returns a model with the hyperparameters inputted in :func:
        `get_parameters`.

        Returns:
            A Random Forest Regression model
        """
        params = self.get_parameters(**self.kwargs)
        return RandomForestRegressor(**params)


class GBT(Model):
    """Subclasses base :class: `Model` to initialize a Gradient Boost Tree
    Regressor.
    """
    def __init__(self, input_cols, **kwargs):
        """Initialises the class.

        Args:
            input_cols(list):
                A list of all the input column names
            **kwargs:
                keyword arguments of user defined parameters
        """
        self.input_cols = input_col
        self.kwargs = kwargs
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
                          "maxDepth": 4,
                          "maxIter": 10,
                          "maxBins": 32}
        parameter_dict.update(**user_params)
        return parameter_dict

    def model_define(self):
        """Returns a model with the hyperparameters inputted in :func:
        `get_parameters`.

        Returns:
            A Gradient Boosting Tree Regression model
        """
        params = self.get_parameters(**self.kwargs)
        return GBTRegressor(**params)
