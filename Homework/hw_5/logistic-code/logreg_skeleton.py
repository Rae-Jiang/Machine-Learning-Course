
def f_objective(theta, X, y, l2_param=1):
    '''
    Args:
        theta: 1D numpy array of size num_features
        X: 2D numpy array of size (num_instances, num_features)
        y: 1D numpy array of size num_instances
        l2_param: regularization parameter

    Returns:
        objective: scalar value of objective function
    '''
    import numpy as np
    res = 0
    for i in range(len(y)):
        res += np.logaddexp(0,(-y[i]*np.dot(theta,X[i])))

    return res/len(y) + l2_param * np.dot(theta,theta)

def fit_logistic_reg(X, y, objective_function, l2_param=1):
    '''
    Args:
        X: 2D numpy array of size (num_instances, num_features)
        y: 1D numpy array of size num_instances
        objective_function: function returning the value of the objective
        l2_param: regularization parameter

    Returns:
        optimal_theta: 1D numpy array of size num_features
    '''
    import numpy as np
    from scipy.optimize import minimize
    MIN = np.amin(X,axis=0)
    MAX = np.amax(X,axis=0)
    X_normalized = (X - MIN)/(MAX - MIN)
    bias = np.ones(X.shape[0])
    X = np.c_[X_normalized, bias]
    y[y==0] = -1
    
    theta = np.zeros(X.shape[1])
    
    def transform_func(theta):   #func in minimize only contain one attribute theta
        return objective_function(theta, X, y, l2_param)
    
    optimal_theta = minimize(transform_func,theta)
    return optimal_theta

                           
                            
        