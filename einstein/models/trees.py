"""
A Script to implement Regression using Trees
"""


from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.regression import RandomForestRegressor,\
                                  DecisionTreeRegressor,\
                                  GBTRegressor
from pyspark.sql.session import SparkSession
from pyspark import SparkConf, SparkContext
from pyspark.ml import Pipeline
from base import Model


class DecisionTree(Model):
    """Subclasses base :class: `Model` to initialize a Decision Tree Regressor
    """
    def get_parameters(self):
        """Declares a dictionary of hyperparameters for Regression.

        Returns:
            A  dictionary containing all the parameters
        """
        parameter_dict = {"featuresCol": "scaledFeatures",
                          "maxDepth": 4,
                          "maxBins": 32}
        return parameter_dict

    def model_define(self):
        """Returns a model with the hyperparameters inputted in :func:
        `get_parameters`.

        Returns:
            A regression model
        """
        params = self.get_parameters()
        return DecisionTreeRegressor(**params)

    def flow(self):
        """Defines a pipeline that tracks the sequence of transformations to
        perform on the data.

        Returns:
            Pipeline
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


class RF(Model):
    """Subclasses base :class: `Model` to initialize a Random Forest Regressor
    """
    def get_parameters(self):
        """Declares a dictionary of hyperparameters for Regression.

        Returns:
            A  dictionary containing all the parameters
        """
        parameter_dict = {"featuresCol": "scaledFeatures",
                          "numTrees": 20,
                          "maxDepth": 4,
                          "maxBins": 32}
        return parameter_dict

    def model_define(self):
        """Returns a model with the hyperparameters inputted in :func:
        `get_parameters`.

        Returns:
            A regression model
        """
        params = self.get_parameters()
        return RandomForestRegressor(**params)

    def flow(self):
        """Defines a pipeline that tracks the sequence of transformations to
        perform on the data.

        Returns:
            Pipeline
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


class GBT(Model):
    """Subclasses base :class: `Model` to initialize a Gradient Boost Tree
    Regressor.
    """
    def get_parameters(self):
        """Declares a dictionary of hyperparameters for Regression.

        Returns:
            A  dictionary containing all the parameters
        """
        parameter_dict = {"featuresCol": "scaledFeatures",
                          "maxDepth": 4,
                          "maxIter": 10,
                          "maxBins": 32}
        return parameter_dict

    def model_define(self):
        """Returns a model with the hyperparameters inputted in :func:
        `get_parameters`.

        Returns:
            A regression model
        """
        params = self.get_parameters()
        return GBTRegressor(**params)

    def flow(self):
        """Defines a pipeline that tracks the sequence of transformations to
        perform on the data.

        Returns:
            Pipeline
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
