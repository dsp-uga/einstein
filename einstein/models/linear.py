"""
A script to implement regression models
"""
from pyspark.ml.regression import LinearRegression
from pyspark.ml import Pipeline
from pyspark.ml.feature import Normalizer

class LinearRegression(model):
    """A class for linear regression extending an abstract class model.
    Args:
        model: an Abstract Class defined in base.py
    """

    def __init__(self,something):
        print("Init called.")
        self.something = something

    def get_parameters(self):
        """A method that defines a dictionary of parameters for regression model
        
        Returns:
            A  dictionary containing all the parameters
        """
        # TODO: to create variables for these parameters that can be initialized in __init__ method.

        return {'featuresCol' : 'features', 'labelCol':'label', 'predictionCol':'prediction', 'maxIter': 20, 'regParam':0.5, 'elasticNetParam':0.5, \
                'tol':1e-06, 'fitIntercept': True, 'standardization': True, 'solver':'auto', 'weightCol':None, 'aggregationDepth':2, 'loss':'squaredError', 'epsilon':1.35}

    def model_define(self,parameter_dict):
        """A method which defines the linear regression model
        
        Returns:
            A linear regression model 
        """
        lr = LinearRegression(**parameter_dict)
        return lr

    def flow(self):
        """A method that defines a pipeline
        
        Returns:
            A pipeline that the model needs to follow
        """
        # TODO: define boolean global variables for each transformation and estimation
        # TODO: make if else statements based on the booleans.

        normalizer = Normalizer(inputCol="features", outputCol="normFeatures", p=1.0)
        pipeline = Pipeline(stages=[normalizer])
        return pipeline