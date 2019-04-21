"""
A script to implement regression models
"""
from pyspark.ml.regression import LinearRegression
from pyspark.ml import Pipeline
from pyspark.ml.feature import Normalizer
from pyspark.sql.session import SparkSession
from pyspark import SparkConf, SparkContext
from base import Model
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.evaluation import RegressionEvaluator


class LinearRegression(model):
    """
    A class for linear regression extending an abstract class model.
    """

    def __init__(self, something):
        print("Init called.")
        self.something = something

    def get_parameters(self):
        """
        A method that defines a dictionary of parameters for regression model
        
        Returns:
            A  dictionary containing all the parameters
        """
        # TODO: to create variables for these parameters that can be initialized in __init__ method.
        return {'maxIter': 20,'regParam': 0.5, 'elasticNetParam': 0.5, 'tol': 1e-06, 'loss': 'squaredError',\
                'epsilon': 1.35}

    def model_define(self):
        """
        A method which defines the linear regression model
        
        Returns:
            A linear regression model 
        """
        params = self.get_parameters()
        lr = LinearRegression(**params)
        return lr

    def flow(self):
        """
        A method that defines a pipeline
        
        Returns:
            A pipeline that the model needs to follow
        """
        assembler = VectorAssembler(inputCols=["cylinders",
                                               "displacement",
                                               "horsepower",
                                               "weight",
                                               "acceleration",
                                               "model year",
                                               "origin"],
                                    outputCol="features")
        scaler = StandardScaler(inputCol="features",
                                outputCol="scaledFeatures",
                                withStd=True, withMean=True)
        model = self.model_define()
        pipeline = Pipeline(stages=[assembler, scaler, model])
        return pipeline


class RidgeRegression(LinearRegression):

    def get_parameters(self):
        """
               A method that defines a dictionary of parameters for ridge regression model by using L2 norm

               Returns:
                   A  dictionary containing all the parameters
               """
        # TODO: to create variables for these parameters that can be initialized in __init__ method.
        return {'maxIter': 20, 'regParam': 0.5, 'elasticNetParam': 0.0, 'tol': 1e-06, 'loss': 'squaredError',\
                'epsilon': 1.35}


class LassoRegression(LinearRegression):

    def get_parameters(self):
        """
               A method that defines a dictionary of parameters for lasso regression model by using L1 norm

               Returns:
                   A  dictionary containing all the parameters
               """
        # TODO: to create variables for these parameters that can be initialized in __init__ method.
        return {'maxIter': 20, 'regParam': 0.5, 'elasticNetParam': 1.0, 'tol': 1e-06, 'loss': 'squaredError', \
                'epsilon': 1.35}

