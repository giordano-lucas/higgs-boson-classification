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

# create optimal model (found by the script 'run_cv.py')
best_param = {'degree':18,'lambda':3.1622776601683795e-09}
expanser = PolynomialExpansion(
    best_param['degree'],
    with_interractions=True,
    with_scaler=True)
# expansion + scaling
tX = expanser.expand(X) 
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