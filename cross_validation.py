# Function to split data indices
# num_examples: total samples in the dataset
# k_fold: number fold of CV
# returns: array of shuffled indices with shape (k_fold, num_examples//k_fold)
import numpy as np
from itertools import product
from implementations import *
from build_polynomial import PolynomialExpansion
from Scaler import StandardScaler

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
    #print("*********************************")
    #******************
    # use one split to test 
    test_ind = k_fold_ind[k]
    # use k-1 split to train
    train_splits = [i for i in range(k_fold_ind.shape[0]) if i is not k]
    train_ind    = k_fold_ind[train_splits,:].reshape(-1)
    # get train and val
    cv_X_tr = X_t[train_ind,:]
    cv_y_tr = y[train_ind]
    cv_X_te = X_t[test_ind,:]
    cv_y_te = y[test_ind]
    # normalize 
    scaler = StandardScaler(has_bias=True)
    cv_X_tr = scaler.fit(cv_X_tr)
    cv_X_te = scaler.transform(cv_X_te)
    m = np.mean(cv_X_tr,axis=0)
    print(np.all(m[1:]<1e-4))
    #print("---------------------------------")
    #fit on train set
    w,loss_tr = ridge_regression(cv_y_tr,cv_X_tr,params['lambda'])
    #get loss for val
    loss_te = compute_loss(cv_y_te,cv_X_te,w)
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
        loss = np.zeros(k_fold)
        for k in range(k_fold):
            loss[k] = do_cross_validation(k,fold_ind,X,y,p,_CACHE_)
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