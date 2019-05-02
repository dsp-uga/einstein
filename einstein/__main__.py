"""
This script is the `main` script, which is the entry point to the :mod:
`einstein`. The user can input a variety of arguments to run the models
for analyzing and experimenting with solar data. Evaluation metrics such
as 'r-Squared', 'Mean Absolute Error' and 'Root Mean Squared Error' are
returned for analysis, on running the regression models, and plots of
'GHI', 'DHI' and 'DNI' versus the timestamps are returned for analysis,
on running the `PVLibModel`.

Author:
-----------
Aashish Yadavally
"""
import findspark
findspark.init()

import argparse
import subprocess
import pandas as pd
from einstein.dataset import Loader
from einstein.models.pvlib import PVLibModel
from einstein.models.linear import (LinearRegressor, RidgeRegressor,
                                    LassoRegressor)
from einstein.models.trees import DTRegressor, RFRegressor, GBTreeRegressor
from einstein.utils.summary_table import draw


def get_parser():
    """Creates a new argument parser

    Returns:
        parser (argparse.ArgumentParser)
            Parser contains user-defined arguments
    """
    parser = argparse.ArgumentParser(description='Solar Irradiance Prediction')
    parser.add_argument('--model', dest='model', default='mlr', type=str,
                        choices=['mlr', 'rr', 'lr', 'dt', 'rf', 'gbt',
                        'pvlib'], help='PVLib Forecast Model, or Models for\
                        prediction: Linear Regression, Ridge Regression,\
                        Lasso Regression, Decision Trees, Random Forests,\
                        Gradient Boost Trees')
    parser.add_argument('--grid', dest='grid', default='3', type=str,
                        choices=['1', '3', '5'], help='1 -> (1, 1) ;\
                        3 -> (3, 3); 5 -> (5, 5)\nGrid Size - Grid sizes\
                        around ATHENS location')
    parser.add_argument('--bucket', dest='bucket', type=str,
                        default='gs://uga_dsp_sp19/', help='Google Storage\
                        Bucket address containing the dataset')
    parser.add_argument('--year', dest='year', default='2017', type=str,
                        choices=['2017', '2018', '2017_2018'],
                        help='NAM-NMM data year range')
    parser.add_argument('--target_hr', dest='target_hr', default=1, type=int,
                        choices=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                                 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
                        help='Prediction for hour offset in the reftime\
                        dimension')
    parser.add_argument('--maxDepth', dest='maxDepth', type=int, help='Maximum\
                        number of levels in each decision tree - parameter for\
                        Decision Trees, Random Forests, Gradient Boost Trees')
    parser.add_argument('--maxBins', dest='maxBins', type=int,
                        help='Number of bins used for discretizing continuous\
                        features - parameter for Decision Trees, Random\
                        Forests, Gradient Boost Trees')
    parser.add_argument('--numTrees', dest='numTrees', type=int, help='Number\
                        of trees in the ensemble - parameter for Random\
                        Forests.')
    parser.add_argument('--maxIter', dest='maxIter', type=int, help='Maximum\
                        number of iterations - parameter for Linear\
                        Regresion, Ridge Regression, Lasso Regression,\
                        Gradient Boost Trees')
    parser.add_argument('--tol', dest='tol', type=float, help='Convergence\
                        tolerance for iterative algorithms - parameter for\
                        Linear Regression, Ridge Regression, Lasso Regression')
    parser.add_argument('--regParam', dest='regParam', type=float,
                        help='Regularization parameter - parameter for Linear\
                        Regression, Ridge Regression, Lasso Regression')
    parser.add_argument('--loss', dest='loss', type=str, help='Loss function \
                        to be optimized - parameter for Linear Regression,\
                        Ridge Regression, Lasso Regression.')
    parser.add_argument('--epsilon', dest='epsilon', type=float, help='The \
                        shape parameter to control the amount of robustness\
                        (must be > 1.0) - parameter for Linear Regression,\
                        Ridge Regression, Lasso Regression')
    parser.add_argument('--fm', dest='forecast_model', type=str, help='PVLib\
                        Forecast Model', default='NAM', choices=['NAM', 'RAP',
                        'GFS', 'NDFD', 'HRRR'])
    parser.add_argument('--start', dest='start', type=pd.Timestamp,
                        help='Start of PVLib Forecast')
    parser.add_argument('--stop', dest='stop', type=pd.Timestamp, help='End\
                        of PVLib Forecast')
    parser.add_argument('--lat', dest='lat', type=float, help='Latitude')
    parser.add_argument('--lon', dest='lon', type=float, help='Longitude')
    parser.add_argument('--test', dest='test', type=bool, default=False,
                        choices=[True, False], help='To run the test suite')
    return parser


def run(args=None):
    '''Main entry point for :mod: `einstein` project

    Arguments:
        args (list):
            A list of arguments as input in the command line at the time of
            execution of the project. `None` lets it use sys.argv
    '''
    parser = get_parser()
    args = parser.parse_args()

    if args.test:
        print('Running Test Suite...')
        subprocess.call("python -m pytest", shell=True)
    elif args.model == "pvlib":
        pvlib_model = PVLibModel(model_name=args.forecast_model, lat=args.lat,
                                 lon=args.lon, start=args.start,
                                 stop=args.stop)
        pvlib_model.plot()
    else:
        filename = f'{args.year}_{args.grid}.csv'

        loader = Loader(target_hour=args.target_hr, bucket=args.bucket,
                        filename=filename)
        df = loader.load_data().repartition(48)
        input_cols = loader.input_cols
        if args.model == "mlr":
            model_name = 'Linear Regression'
            regressor = LinearRegressor(input_cols, maxIter=args.maxIter,
                                        regParam=args.regParam, tol=args.tol,
                                        loss=args.loss, epsilon=args.epsilon)
        elif args.model == "rr":
            model_name = 'Ridge Regression'
            regressor = RidgeRegressor(input_cols, maxIter=args.maxIter,
                                       regParam=args.regParam, tol=args.tol,
                                       loss=args.loss, epsilon=args.epsilon)
        elif args.model == "lr":
            model_name = 'Lasso Regression'
            regressor = LassoRegressor(input_cols, maxIter=args.maxIter,
                                       regParam=args.regParam, tol=args.tol,
                                       loss=args.loss, epsilon=args.epsilon)
        elif args.model == "dt":
            model_name = 'Decision Tree'
            regressor = DTRegressor(input_cols, maxDepth=args.maxDepth,
                                    maxBins=args.maxBins)
        elif args.model == "rf":
            model_name = 'Random Forest'
            regressor = RFRegressor(input_cols, numTrees=args.numTrees,
                                    maxDepth=args.maxDepth,
                                    maxBins=args.maxBins)
        elif args.model == "gbt":
            model_name = 'Gradient Boost Tree'
            regressor = GBTreeRegressor(input_cols, maxDepth=args.maxDepth,
                                        maxIter=args.maxIter,
                                        maxBins=args.maxBins)

        train_df, test_df = df.randomSplit([0.9, 0.1], seed=100)
        # Repartitioning the data
        train_df.repartition(48)
        test_df.repartition(48)
        r2, mae, rmse = regressor.fit_transform(train_df, test_df)

        metrics = {'r-Squared': r2, 'Mean Absolute Error': mae,
                   'Root Mean Squared Error': rmse}
        # Print the Regression Statistics Summary
        draw(args.year, args.target_hr, args.grid, model_name, metrics)


if __name__ == '__main__':
    run()
