import numpy as np
from implementations import *
import datetime
from proj1_helpers import *
from missing_values import *
from cross_validation import *
# Directory where the data can be found
DIRECTIORY = 'data/'
# data loading
DATA_TRAIN_PATH = DIRECTIORY + 'train.csv' 
y, X, ids = load_csv_data(DATA_TRAIN_PATH)
# missing value interpolation
interpolator = LinearInterpolator()
X = interpolator.interpolate(X)
# hyperparameter search space
degrees = np.arange(1,9)
lambdas = np.logspace(-4, 2, 10)
params={'degree':degrees,'lambda':lambdas}
# cross validation based on accuracy instead of loss
def error(y,X,w):
    """mean prediction error"""
    pred = predict_labels(w,X)
    return np.mean(y!=pred)
# call to the grid search function
best_param = grid_search_cv(
    params,
    X,y,
    k_fold=10,
    loss_ft=error)
# create optimal model 
expanser = PolynomialExpansion(
    best_param['degree'],
    with_interractions=True,
    with_scaler=True)
tX = expanser.expand(X) # expansion + scaling
# compute the weights
weights,loss_tr = ridge_regression(y,tX,best_param['lambda'])
# load test set
DATA_TEST_PATH = DIRECTIORY + 'test.csv' 
y_sub, X_sub, ids_sub = load_csv_data(DATA_TEST_PATH)
# apply the same tranformation pipeline to the test set
Xt_sub = interpolator.interpolate(X_sub)
Xt_sub = expanser.expand(Xt_sub)
# predict the labels
y_pred = predict_labels(weights, Xt_sub)
# create submission
OUTPUT_PATH = DIRECTIORY + 'submission.csv'
create_csv_submission(ids_sub, y_pred, OUTPUT_PATH)