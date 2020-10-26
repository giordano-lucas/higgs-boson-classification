# ReadMe Team Pleutre
***
***
## Preliminary steps
1. The script 'run.py' currently assumes that the dataset is located in a directory named 'data'. However, this can easily be changed by updating line 8 in the script: ```DIRECTIORY = 'data/' ```
2. The runnable script requires the following list of files to be placed in the current directiory : 
        
    * missing_values.py
    * Scaler.py
    * build_polynomial.py
    * cross_validation.py 

3. Note that in the context of logistic regression, the methods computing the loss and gradient all suppose that the variable ```y``` is such that $\forall_i \; y_i \in \{0,1\}$. Since the higgs dataset labels are in the set $\{-1,1\}$, we decided to map $-1$ into $0$ in the following functions. Note that they can also be called with labels in $\{0,1\}$ and they will work correcly
```python
def logistic_regression(y,tx,initial_w,max_iters,gamma):
    """Logistic regression with gradient descent"""
    ...

def reg_logistic_regression(y,tx,lambda_,initial_w,max_iters,gamma):
    """Logistic regression with stochastic gradient descent"""
    ...
```

##  Data preparation

After loading the dataset, our first transformation will be to interpolate the missing values. 
In the file ```missing_values.py```, we can find two classes in charge of this process: 

* MeanInterpolator: for each feature, the missing values are interpolated to 
    the mean of the feature. 
* LinearInterpolator: For each feature $c$ that contains missing value the we train a simple least squares task to predict the missing values. We set the target variable $y$ to be equal to $c$, letting $X$ to be all the columns of the higgs dataset except $c$. In this process we droped the actual label column in the dataset so that the procedure could also be applied to fill the missing values for the submission dataset, for which the labels are not available. The next step is to split $y$ and $X$ into a training and test set according to the missing values in $c$. To do so, we define the following mask: $mask_i = I\{y_i = -999\}$. We have:

     * $y_{test} = y[mask]$
     * $y_{train} = y[~mask]$
     * $X_{test} = X[mask]$
     * $X_{train} = X[~mask]$ 
We can then train the model and interpolate the values for the missing values in $c$ by predicting $y_{test}$. 
Note that the class keeps track of the fitted model so that the exact same interpolation can be applied to the submission set. 

In the ```run.py``` script, we choose a LinearInterpolator:
```python 
interpolator = LinearInterpolator()
X = interpolator.interpolate(X)
```

We also need to normalise the features in the data preparation phase. However, since we also use a polynomial feature expansion, the idea presented below will only be applied after the expansion.

All our code related to normalisation can be found in the file ```Scaler.py```. We defined two normalisation procedures:

* StandardScaler : standard Scaler that returns a matrix with 0-mean columns and 1-std columns
* MinMaxScaler : scaler that returns a matrix scaler where the columns were normalised according to the formula : $(x-min)/(max-min)$

Similarly to the interpolator classes, the scaler classes stores as attributes the parameters of the scaler leared on the training set so that they can be reused on the submission set.

In the code, we use a Standard Scaler but this is not immediately visible in the script since this is embedded in the polynomial expansion class.

##  Feature generation

All our code related to the polynomial feature expansion was placed in the file ```build_polynomial.py```, under the class name ```PolynomialExpanser```. 

Our feature expansion pipeline for a given degree $d$ is the following:
* Add a column of ones (bias)
* Construct $X^i, \; \forall_{0<i\leq d}$ and concatenate the result along the column axis. 
* Add all interaction terms of degree $2$. 

This is summarised in the following piece of code 
```python
def expand(self,X):
        """
        @param X       : (numpy array) input data
        @param degree  : (int) degree of the feature expansion
        @return X_poly : (numpy array) the expanded version of X
        """
        X_poly = build_poly(X,self.degree)    # add non interaction terms
        if self.with_interractions:
            interractions=get_interactions(X) # interractions of X
            X_poly= np.concatenate((X_poly, interractions), axis=1) # add the interraction terms to the final result
        if self.with_scaler:                  # if scaling is needed
            X_poly = self.scale(X_poly)
        X_poly = add_bias(X_poly)             # add a bias column to X_poly  
        return X_poly
```
we can see that we also normalise the expanded dataset using a standard scaler.

##  Cross-validation steps

We chose to implement a $k$-fold cross validation procedure. The code can be found in the file ```cross_validation.py``` in the method:
```python
def grid_search_cv(params,X,y,k_fold=15,model=cross_val_ridge_regression,loss_ft=rmse):
    ...
```
The proposed code is fairly general and was made to minimise the amount of code that needs to be changed when the model is changed. In that sense, we implemented a ```ParameterGrid``` object that represents the cartesian product of a set of hyperparameters. It contains a generate method that is supposed to be used in a for loop to get all the combinations of hyperparameters. The next step in this 'general' approach was to parametrise the ```grid_search_cv``` function with a model and a loss, so that the code is independent of the model used to compute the weights. After having trained $k$ different model, we store in an array, the mean loss induced by those model to obtain a more reliable estimate of the true value of the test loss. This array is then passed to the method 
```python
def get_best_params(grid_mean,param_grid,verbose=True):
    ...
```
which returns the set of hyperparameters associated with the minimum value of the array of mean losses. This is the result of our cross validation algorithm.

The method :
```python
def grid_search_cv(params,X,y,k_fold=15,model=cross_val_ridge_regression,loss_ft=rmse):
    ...
```
calls itself the method 
```python
def do_cross_validation(k,k_fold_ind,X,y,params,_CACHE_,model,loss_ft):
    ...
```
which is responible of training a model for a specific combination of hyperparameter and a specific fold of the dataset. This method also has the option to cache some results in order to speed up the computation. In our code this is used to avoid the costly computation of the feature expansion for a given degree when it was already computed before.

Regarding data transformation this method performs the train/test split, the polynomial expansion and the normalisation.

In the ```run.py``` script the code associated to the cross validation is the following
```python
# hyperparameter search space
degrees = np.arange(1,12)
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
    k_fold=15,
    model=cross_val_ridge_regression,
    loss_ft=error)
```

We defined the ```error``` function as the complement of the accuracy. Since we are in a classification setting and that our model is judged on its accuracy, it makes sense to choose the best set of hyperparameters using this metric instead of the root mean square error. In the code, our search space and the number of folds can be found.

## Reproducibility

The code provided in the script ```run.py``` will produce 
