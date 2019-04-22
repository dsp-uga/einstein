"""
A Script to implement a base abstract class which will be used
by all regression models
"""


from abc import ABC, abstractmethod
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml import Pipeline


class Model(ABC):
    """A class extending an abstract class 'ABC'.
    A class with regress_model and flow abstract methods and

    Args:
        ABC: an Abstract Base Class
    """

    @abstractmethod
    def get_parameters(self):
        """An abstract method which needs to be declared in subclasses
        This method is used to declare different parameters for methods
        in subclasses

        Returns:
            A  dictionary containing all the parameters
        """
        pass

    @abstractmethod
    def model_define(self):
        """An abstract method which needs to be declared in subclasses.
        This method is used to declare different regression models
        in subclasses

        Returns:
            A regression model
        """
        pass

    def flow(self):
        """Defines a pipeline that tracks the sequence of transformations to
        perform on the data.

        Returns:
            Pipeline
        """
        assembler = VectorAssembler(inputCols=self.input_cols,
                                    outputCol="features")
        scaler = StandardScaler(inputCol="features",
                                outputCol="scaledFeatures",
                                withStd=True, withMean=True)
        model = self.model_define()
        pipeline = Pipeline(stages=[assembler, scaler, model])
        return pipeline

    def fit_transform(self, train_data, test_data):
        """A method to train the model and do predictions.
        a function for calculating the metric value is called

        Args:
            train_data: the data on which the model should be trained
            test_data: the data on which the predictions are to be made
        Returns:
            A list containing metric values ["r2", "mae", "rmse"]
        """
        pipeline = self.flow()
        model = pipeline.fit(train_data)
        predictions = model.transform(test_data)
        predictions = predictions['prediction', 'label']
        return self.get_accuracy(predictions)  

    def get_accuracy(self, predictions):
        """A method to calculate the selected metric value

        Args:
            predictions: a dataframe containing prediction and label
        Returns:
            A list containing metric values ["r2", "mae", "rmse"]
        """
        metric_list = []
        for metric_name in self.metrics:
            evaluator = RegressionEvaluator(labelCol='label',
                                            predictionCol='prediction',
                                            metricName=metric_name)
            metric_answer = evaluator.evaluate(predictions)
            metric_list.append(metric_answer)
        return metric_list
