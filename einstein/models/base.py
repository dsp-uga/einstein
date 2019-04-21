"""
A Script to implement a base abstract class which will be used
by all regression models
"""


from abc import ABC, abstractmethod


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

    @abstractmethod
    def flow(self):
        """An abstract method which needs to be declared in subclasses
        This method is used to declare different Pipeline
        in subclasses
        
        Returns:
            A pipeline that the model needs to follow
        """
        pass

    def fit_transform(self, train_data, test_data):
        """A method to train the model and do predictions.
        a function for calculating the metric value is called

        Args:
            train_data: the data on which the model should be trained
            train_data: the data on which the predictions are to be made
        Returns:
            Metric value
        """
        pipeline = flow()
        model = pipeline.fit(train_data)
        predictions = model.transform(test_data)
        predictions = predictions['prediction', 'label']
        return get_accuracy(predictions)

    def get_accuracy(self, predictions, metric_name = 'rmse'):
        """A method to calculate the selected metric value

        Args:
            predictions: a dataframe containing prediction and label
        Returns:
            Metric value
        """
        evaluator = RegressionEvaluator(labelCol=label,
                                        predictionCol=prediction,
                                        metricName = metric_name)
        metric_answer = evaluator.evaluate(predictions)
        return metric_answer
