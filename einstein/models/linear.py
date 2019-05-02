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

from pyspark.ml.regression import LinearRegression
from einstein.models.base import Model
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder



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
        self.kwargs = {k: v for k, v in kwargs.items() if v is not None}
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
        for k, v in parameter_dict.items():
            print(f'{k} : {v}')
        return parameter_dict

    def model_define(self):
        """A method which defines the linear regression model

        Returns:
            A linear regression model
        """
        params = self.get_parameters(**self.kwargs)
        linear_reg = LinearRegression(**params)
        return linear_reg

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
        evaluator = RegressionEvaluator(labelCol='label',
                                        predictionCol='prediction',
                                        metricName="rmse")
        paramGrid = ParamGridBuilder() \
            .addGrid(lr.regParam, [0.5]) \
            .build()
        crossval = CrossValidator(estimator=pipeline,
                                  estimatorParamMaps=paramGrid,
                                  evaluator=evaluator,
                                  numFolds=4)
        cvmodel = crossval.fit(training)
        predictions = cvmodel.transform(test_data)
        predictions = predictions['prediction', 'label']
        cv_metric_rmse = evaluator.evaluate(predictions)

        model = pipeline.fit(train_data)
        predictions = model.transform(test_data)
        predictions = predictions['prediction', 'label']
        metric_list=self.get_accuracy(predictions)
        metric_list.append(cv_metric_rmse)
        return metric_list


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
        for k, v in parameter_dict.items():
            print(f'{k} : {v}')
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
        for k, v in parameter_dict.items():
            print(f'{k} : {v}')
        return parameter_dict
