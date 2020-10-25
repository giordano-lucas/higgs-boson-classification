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

3. Note that in the context of logistic regression, the methods computing the loss and gradient all suppose that the variable ```y``` is such that  $\forall_i \; y_i \in \{0,1\}$. Since the higgs dataset labels are in the set $\{-1,1\}$, we decided to map $-1$ into $0$ in the following functions. Note that they can also be called with labels in $\{0,1\}$ and they will work correcly
```python
def logistic_regression(y,tx,initial_w,max_iters,gamma):
    """Logistic regression with gradient descent"""
    ...

def reg_logistic_regression(y,tx,lambda_,initial_w,max_iters,gamma):
    """Logistic regression with stochastic gradient descent"""
    ...
```

##  Data preparation

##  Feature generation
##  Cross-validation steps