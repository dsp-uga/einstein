"""
A Script to implement a base abstract class which will be used
by all regression models
"""


from abc import ABC, abstractmethod


class DefaultReg(ABC):
    """
    A class extending an abstract class 'ABC'.
    A class with regress_model and flow abstract methods and

    Args:
        ABC: an Abstract Base Class
    """
    @abstractmethod
    def regress_model(self):
        """
        An abstract method which needs to be declared in subclasses.
        This method is used to declare different regression models
        in subclasses
        """
        pass

    @abstractmethod
    def flow(self):
        """
        An abstract method which needs to be declared in subclasses
        This method is used to declare different Pipeline
        in subclasses
        """
        pass

    def fit_transform(self, train_data, test_data):
        """
        A method to train the model and do predictions.
        a function for calculating the rmse value is called

        Args:
            train_data: the data on which the model should be trained
            train_data: the data on which the predictions are to be made

        Returns:
            RMSE value
        """
        pipeline = flow()
        model = pipeline.fit(train_data)
        predictions = model.transform(test_data)
        predictions = predictions['prediction', 'label']
        return get_accuracy(predictions)

    def get_accuracy(self, predictions):
        """
        a method to calculate the rmse value

        Args:
            predictions: a dataframe containing prediction and label
        Returns:
            RMSE value
        """
        evaluator = RegressionEvaluator(labelCol=label,
                                        predictionCol=prediction)
        rmse = evaluator.evaluate(predictions)
        return rmse
