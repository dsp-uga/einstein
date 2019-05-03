"""
This script defines an abstract base :class: `Model`,
with abstract methods :func: `get_parameters` and :func:
`model_define`, which is sub-classed by the Linear and Tree
Regression models defined in :mod: `einstein`.

Author:
----------
Anirudh Kumar Maurya Kakarlapudi
"""

import findspark
findspark.init()

from abc import ABC, abstractmethod
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.tuning import CrossValidator
from pyspark.ml import Pipeline


class Model(ABC):
    """An abstract class with :func: `regress_model` and :func: `flow`
    abstract methods, which fits a regression model on the train data
    and predicts the targets for the test data
    """
    @abstractmethod
    def get_parameters(self):
        """An abstract method which needs to be declared in subclasses
        This method is used to declare different parameters for methods
        in the derived classes

        Returns:
            (dict):
            Dictionary containing all the parameters
        """
        pass

    @abstractmethod
    def model_define(self):
        """An abstract method which needs to be declared in subclasses.
        This method is used to declare different regression models
        in the derived classes

        Returns:
            A regression model
        """
        pass

    def flow(self):
        """Defines a pipeline that tracks the sequence of transformations to
        perform on the data.

        Returns:
            (psypark.ml.Pipeline)
                Pipeline containing sequence of data transformations
        """
        assembler = VectorAssembler(inputCols=self.input_cols,
                                    outputCol="features")
        scaler = StandardScaler(inputCol="features",
                                outputCol="scaledFeatures",
                                withStd=True, withMean=True)
        self.model_definition = self.model_define()
        pipeline = Pipeline(stages=[assembler, scaler, self.model_definition])
        return pipeline

    def fit_transform(self, train_data, test_data):
        """Fits a model using cross-validation and predicts targets for
        test data.

        Args:
            train_data (pyspark.DataFrame):
                The data on which the model should be fit
            test_data (pyspark.DataFrame):
                The data on which the predictions are to be made
        Returns:
            (list):
                List containing metric values ["r2", "mae", "rmse"]
        """
        pipeline = self.flow()
        param_grid = self.get_param_grid(self.model_definition)
        cross_validator = CrossValidator(estimator=pipeline,
                                         estimatorParamMaps=param_grid,
                                         evaluator=self.get_evaluator(),
                                         numFolds=5)
        cv_model = cross_validator.fit(train_data)
        predictions = cv_model.transform(test_data)
        predictions = predictions['prediction', 'label']
        return self.get_accuracy(predictions)

    def get_evaluator(self, metric='mae'):
        """Defines a regression evaluator on the given metric

        Args:
            metric (string):
                Metric to git the regression evaluator on - can be one of
                'mae', 'r2', 'rmse'
        Returns:
            evaluator (pyspark.ml.evaluation.RegressionEvaluator)
                Regression evaluator fitted on the user-defined metric
        """
        if metric not in self.metrics:
            print("Invalid Metric: One of 'r2', 'mae' and 'rmse' expected.")
        else:
            evaluator = RegressionEvaluator(labelCol='label',
                                            predictionCol='prediction',
                                            metricName=metric)
        return evaluator

    def get_accuracy(self, predictions):
        """Calculates the selected metric value

        Args:
            predictions (pyspark.DataFrame):
                A dataframe containing prediction and label
        Returns:
            metric_values (list):
                List containing metric values ["r2", "mae", "rmse"]
        """
        metric_values = [self.get_evaluator(metric).evaluate(predictions) \
        for metric in self.metrics]
        return metric_values
