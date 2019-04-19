"""
A script to implement regression models
"""
import org.apache.spark.ml.regression.LinearRegression
from pyspark.ml import Pipeline
from pyspark.ml.feature import Normalizer

class LinearRegression(model): 
	"""A class for linear regression extending an abstract class model.
    Args:
        model: an Abstract Class defined in base.py
    """
	
	def __init__(self, something):
		print("Init called.")
		self.something = something
		
    def get_parameters(self):
         """A method that defines a dictionary of parameters for regression model
        
        Returns:
            A  dictionary containing all the parameters
        """
		# TODO: to create variables for these parameters that can be initialized in __init__ method. 
		parameter_dict={}
		parameter_dict["maxIter"]=10
		parameter_dict["featuresCol"]='features'
		parameter_dict["labelCol"]='label'
		parameter_dict["predictionCol"]='prediction'
		parameter_dict["regParam"]=0.0
		parameter_dict["elasticNetParam"]=0.0
		parameter_dict["tol"]=1e-06
		parameter_dict["fitIntercept"]=True
		parameter_dict["standardization"]=True
		parameter_dict["solver"]='auto'
		parameter_dict["weightCol"]=None
		parameter_dict["aggregationDepth"]=2
		parameter_dict["loss"]='squaredError'
		parameter_dict["epsilon"]=1.35
		return parameter_dict 
		
        
    
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
