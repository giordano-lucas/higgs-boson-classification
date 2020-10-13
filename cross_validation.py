# Function to split data indices
# num_examples: total samples in the dataset
# k_fold: number fold of CV
# returns: array of shuffled indices with shape (k_fold, num_examples//k_fold)
import numpy as np
from itertools import product
from implementations import *
from build_polynomial import PolynomialExpansion
from Scaler import StandardScaler

#================================================================
#======================= Helper Class ===========================
#================================================================
class ParameterGrid:
    def __init__(self,params):
        self.params = params
        self.len    = np.prod(np.array([len(v) for v in params.values()]))
    def __len__(self):
        return self.len
    def get(self,index):
        res = {}
        for i,key in enumerate(self.params.keys()):
            res[key] = self.params[key][index[i]]
        return res

    def generate(self):
        keys = self.params.keys()
        vals = self.params.values()
        for instance in product(*vals):
            yield dict(zip(keys, instance))
#================================================================
#==================== Cross Validation ==========================
#================================================================
def fold_indices(num_examples,k_fold=2):
    ind = np.arange(num_examples)
    split_size = num_examples//k_fold
    #important to shuffle your data
    np.random.shuffle(ind)
    k_fold_indices = []
    # Generate k_fold set of indices
    k_fold_indices = [ind[k*split_size:(k+1)*split_size] for k in range(k_fold)]
    return np.array(k_fold_indices)

def do_cross_validation(k,k_fold_ind,X,y,params,_CACHE_):
    degree = params['degree']
    #*** /!\ CACHE ****
    X_t = None
    if degree in _CACHE_.keys():
        X_t = _CACHE_[degree]
    else: # compute expansion
        expanser = PolynomialExpansion(degree,with_scaler=False)
        X_t      = expanser.expand(X)
        _CACHE_.clear()             # clear dict to save memory
        _CACHE_[degree] = X_t     # add to cache for the next iterations
    #******************
    # use one split to test 
    test_ind  = k_fold_ind[k]
    # use k-1 split to train
    train_ind = k_fold_ind[np.arange(k_fold_ind.shape[0]) != k]
    train_ind = k_fold_ind.reshape(-1)
    # get train and val
    cv_X_tr = X_t[train_ind,:]
    cv_y_tr = y[train_ind]
    cv_X_te = X_t[test_ind,:]
    cv_y_te = y[test_ind]
    # normalize 
    scaler = StandardScaler(has_bias=True)
    cv_X_tr = scaler.fit(cv_X_tr)
    cv_X_te = scaler.transform(cv_X_te)
    #fit on train set
    w,loss_tr = ridge_regression(cv_y_tr,cv_X_tr,params['lambda'])
    
    #get loss for val
    loss_te = loss_least_squares(cv_y_te,cv_X_te,w)
    #print('loss tr',loss_tr, 'loss te',np.sqrt(2*loss_te))
    return np.sqrt(2*loss_te)

def grid_search_cv(params,X,y,k_fold=5):
    '''
    Grid Search Function
    params:{'param1':[1,2,..,4],'param2':[6,7]} dictionary of search params
    k_fold: fold for CV to be done
    fold_ind: splits of training set
    function: implementation of model should return a loss or score
    X,Y: training examples
    '''
    # compute the k_fold indices
    fold_ind = fold_indices(X.shape[0],k_fold=k_fold)
    # generator on the cartesian product of params
    param_grid = ParameterGrid(params) 
    #save the values for the combination of hyperparameters
    grid_mean = np.zeros(len(param_grid)) 
    # cache if needed by do_cross_validation
    _CACHE_ = {}
    for i, p in enumerate(param_grid.generate()):
        loss = []
        for k in range(k_fold):
            loss.append(do_cross_validation(k,fold_ind,X,y,p,_CACHE_))
        grid_mean[i] = np.mean(loss)
        print('Evaluated for {0} : loss = {1}'.format(p,grid_mean[i]))
    # reshape in the proper dimension of search space
    if len(params.keys())>1:
        search_dim = tuple([len(p) for _,p in params.items()])
        grid_mean  = grid_mean.reshape(search_dim)
    return get_best_params(grid_mean,param_grid)

def get_best_params(grid_mean,param_grid,verbose=True):
    #get the best validation score
    best_score = np.min(grid_mean)
    #get degree which gives best score
    ind = np.unravel_index(np.argmin(grid_mean, axis=None), grid_mean.shape)
    best_params = param_grid.get(ind)
    if verbose:
        print('======== Best test loss for parameters {} ========'.format(best_params))
        print('======== Best test loss score {} ========'.format(best_score))
    return best_params

#================================================================
#=================== Non Cross Validation =======================
#================================================================

def train_test_split(X,y,train_ratio=0.67,seed=43):
    """
    split the dataset based on the split ratio. If ratio is 0.8 
    you will have 80% of your data set dedicated to training 
    and the rest dedicated to testing
    """
    np.random.seed(seed)                # set seed
    N = X.shape[0]                      # number of data points 
    indices = np.arange(N)              # all row indices
    np.random.shuffle(indices)          # shuffle indices
    split_index=int(N*train_ratio)      # last training point 
    mask_tr = indices[:split_index]     # mask for training set
    mask_te = indices[split_index:]     # mask for testing set

    return X[mask_tr],y[mask_tr],X[mask_te],y[mask_te]

def hyper_parameter_optimisation(params,X,y):
    train_set,train_target,test_set,test_target = train_test_split(X,y,train_ratio=0.67)
    # generator on the cartesian product of params
    param_grid = ParameterGrid(params) 
    #save the values for the combination of hyperparameters
    losses = np.zeros(len(param_grid)) 
    #parameters
    old_deg  = None
    expanser = None  # create object for polynomial expansion
    x_tr = None      # expand the training set
    x_te = None      # expand the test set
    y_tr = train_target  # renaming for consistency
    y_te = test_target   # renaming for consistency
    for i, p in enumerate(param_grid.generate()):
        degree = p['degree']
        if old_deg is None or old_deg != degree:
            expanser = PolynomialExpansion(degree,with_interractions=True)  # create object for polynomial expansion
            x_tr = expanser.expand(train_set)       # expand the training set
            x_te = expanser.expand(test_set)        # expand the test set
            old_deg = degree
        #fit on train set
        #w,loss_tr = ridge_regression(y_tr,x_tr,p['lambda'])
        w,loss_tr  = reg_logistic_regression(y_tr,x_tr,
            lambda_=p['lambda'],initial_w=np.zeros((x_tr.shape[1])),max_iters=50,gamma=p['gamma'])
        #get loss for val
        loss = loss_least_squares(y_te,x_te,w)
        losses[i] = np.sqrt(2*loss)
        print('Evaluated for {0} : loss = {1}'.format(p,losses[i]))
    # reshape in the proper dimension of search space
    if len(params.keys())>1:
        search_dim = tuple([len(p) for _,p in params.items()])
        losses  = losses.reshape(search_dim)
    return get_best_params(losses,param_grid)
