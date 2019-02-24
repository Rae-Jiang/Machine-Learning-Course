import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


### Assignment Owner: Tian Wang


#######################################
### Feature normalization
def feature_normalization(train, test):
    """Rescale the data so that each feature in the training set is in
    the interval [0,1], and apply the same transformations to the test
    set, using the statistics computed on the training set.

    Args:
        train - training set, a 2D numpy array of size (num_instances, num_features)
        test - test set, a 2D numpy array of size (num_instances, num_features)

    Returns:
        train_normalized - training set after normalization
        test_normalized - test set after normalization
    """
    # TODO
    #discard features that are constant in the training set
    feature_removal = np.std(train, axis=0) != 0
    train1 = train.transpose()
    train = train1[np.where(feature_removal)].transpose()
    test1 = test.transpose()
    test = test1[np.where(feature_removal)].transpose()
    
    #feature normalization
    MIN = np.amin(train,axis=0)
    MAX = np.amax(train,axis=0)
    train_normalized = (train - MIN)/(MAX - MIN)
    test_normalized = (test - MIN)/(MAX - MIN)
    return train_normalized, test_normalized

#######################################
### The square loss function
def compute_square_loss(X, y, theta):
    """
    Given a set of X, y, theta, compute the average square loss for predicting y with X*theta.

    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        theta - the parameter vector, 1D array of size (num_features)

    Returns:
        loss - the average square loss, scalar
    """
    #TODO
    loss = 0 #Initialize the average square loss
    num_instances = X.shape[0]
    num_features =  X.shape[1]
    factor = np.matmul(X,theta) - y
    loss = 1/num_instances * np.matmul(factor.transpose(),factor).mean()
    return loss

#######################################
### The gradient of the square loss function
def compute_square_loss_gradient(X, y, theta):
    """
    Compute the gradient of the average square loss (as defined in compute_square_loss), at the point theta.

    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        theta - the parameter vector, 1D numpy array of size (num_features)

    Returns:
        grad - gradient vector, 1D numpy array of size (num_features)
    """
    num_instances, num_features = X.shape[0], X.shape[1]
    factor = np.matmul(X,theta) - y
    grad = 2/num_instances * np.matmul(X.transpose(), factor)
    return grad

#######################################
### Gradient checker
#Getting the gradient calculation correct is often the trickiest part
#of any gradient-based optimization algorithm. Fortunately, it's very
#easy to check that the gradient calculation is correct using the
#definition of gradient.
#See http://ufldl.stanford.edu/wiki/index.php/Gradient_checking_and_advanced_optimization
def grad_checker(X, y, theta, epsilon=0.01, tolerance=1e-4):
    """Implement Gradient Checker
    Check that the function compute_square_loss_gradient returns the
    correct gradient for the given X, y, and theta.

    Let d be the number of features. Here we numerically estimate the
    gradient by approximating the directional derivative in each of
    the d coordinate directions:
    (e_1 = (1,0,0,...,0), e_2 = (0,1,0,...,0), ..., e_d = (0,...,0,1))

    The approximation for the directional derivative of J at the point
    theta in the direction e_i is given by:
    ( J(theta + epsilon * e_i) - J(theta - epsilon * e_i) ) / (2*epsilon).

    We then look at the Euclidean distance between the gradient
    computed using this approximation and the gradient computed by
    compute_square_loss_gradient(X, y, theta).  If the Euclidean
    distance exceeds tolerance, we say the gradient is incorrect.

    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        theta - the parameter vector, 1D numpy array of size (num_features)
        epsilon - the epsilon used in approximation
        tolerance - the tolerance error

    Return:
        A boolean value indicating whether the gradient is correct or not
    """
    true_gradient = skeleton_code.compute_square_loss_gradient(X, y, theta) #The true gradient
    num_features = theta.shape[0]
    approx_grad = np.zeros(num_features) #Initialize the gradient we approximate
    #TODO
    grad_checker = True
    for i in range(num_features):
        e_i = np.zeros(num_features)
        e_i[i] = 1
        J1 = skeleton_code.compute_square_loss(X, y, theta+epsilon*e_i)
        J2 = skeleton_code.compute_square_loss(X, y, theta-epsilon*e_i)
        approx_grad[i] = (J1 - J2)/(2*epsilon)
    if np.linalg.norm(approx_grad - true_gradient) > tolerance:
        grad_checker == False
    return grad_checker


#######################################
### Generic gradient checker
def generic_gradient_checker(X, y, theta, objective_func, gradient_func, epsilon=0.01, tolerance=1e-4):
    """
    The functions takes objective_func and gradient_func as parameters. 
    And check whether gradient_func(X, y, theta) returned the true 
    gradient for objective_func(X, y, theta).
    Eg: In LSR, the objective_func = compute_square_loss, and gradient_func = compute_square_loss_gradient
    """
    #TODO


#######################################
### Batch gradient descent
def batch_grad_descent(X, y, alpha=0.1, num_step=1000, grad_check=False):
    """
    In this question you will implement batch gradient descent to
    minimize the average square loss objective.

    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        alpha - step size in gradient descent
        num_step - number of steps to run
        grad_check - a boolean value indicating whether checking the gradient when updating

    Returns:
        theta_hist - the history of parameter vector, 2D numpy array of size (num_step+1, num_features)
                     for instance, theta in step 0 should be theta_hist[0], theta in step (num_step) is theta_hist[-1]
        loss_hist - the history of average square loss on the data, 1D numpy array, (num_step+1)
    """
    num_instances, num_features = X.shape[0], X.shape[1]
    theta_hist = np.zeros((num_step+1, num_features)) #Initialize theta_hist
    loss_hist = np.zeros(num_step+1) #Initialize loss_hist
    theta = np.zeros(num_features) #Initialize theta
    #TODO
    loss_hist[0] = compute_square_loss(X, y, theta) # initial loss
    theta_hist[0] = theta # initial theta
    for i in range(num_step):
        grad = compute_square_loss_gradient(X, y, theta)
        theta = theta - alpha * grad
        theta_hist[i + 1] = theta
        loss = compute_square_loss(X, y, theta)
        loss_hist[i + 1] = loss
    return theta_hist, loss_hist

#######################################
### Backtracking line search
#Check http://en.wikipedia.org/wiki/Backtracking_line_search for details
#TODO
def batch_grad_descent_backtracking(X, y, step=1, alpha=0.25, beta = 0.8,num_step=1000, grad_check=False):
    num_instances, num_features = X.shape[0], X.shape[1]
    theta_hist = np.zeros((num_step+1, num_features)) #Initialize theta_hist
    loss_hist = np.zeros(num_step+1) #Initialize loss_hist
    theta = np.zeros(num_features) #Initialize theta
    
    #TODO
    loss_hist[0] = compute_square_loss(X, y, theta) # initial loss
    theta_hist[0] = theta # initial theta
    for i in range(num_step):
        grad = compute_square_loss_gradient(X, y, theta)
        left = compute_square_loss(X,y,(theta-step*grad)) 
        right = compute_square_loss(X,y, theta) - alpha*step*(np.linalg.norm(grad)**2)
        if left > right:
            step = beta * step
        theta = theta - step * grad
        theta_hist[i + 1] = theta
        loss = compute_square_loss(X, y, theta)
        loss_hist[i + 1] = loss
    return theta_hist, loss_hist


#######################################
### The gradient of regularized batch gradient descent
def compute_regularized_square_loss_gradient(X, y, theta, lambda_reg):
    """
    Compute the gradient of L2-regularized average square loss function given X, y and theta

    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        theta - the parameter vector, 1D numpy array of size (num_features)
        lambda_reg - the regularization coefficient

    Returns:
        grad - gradient vector, 1D numpy array of size (num_features)
    """
    #TODO
    num_instances, num_features =  X.shape[0],X.shape[1]
    factor = np.matmul(X,theta) - y
    grad = 2/num_instances * np.matmul(X.transpose(), factor) + 2 * lambda_reg * theta
    return grad


#######################################
### Regularized batch gradient descent
def regularized_grad_descent(X, y, alpha=0.05, lambda_reg=10**-2, num_step=1000):
    """
    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        alpha - step size in gradient descent
        lambda_reg - the regularization coefficient
        num_step - number of steps to run
    
    Returns:
        theta_hist - the history of parameter vector, 2D numpy array of size (num_step+1, num_features)
                     for instance, theta in step 0 should be theta_hist[0], theta in step (num_step+1) is theta_hist[-1]
        loss hist - the history of average square loss function without the regularization term, 1D numpy array.
    """
    num_instances, num_features = X.shape[0], X.shape[1]
    theta = np.zeros(num_features) #Initialize theta
    theta_hist = np.zeros((num_step+1, num_features)) #Initialize theta_hist
    loss_hist = np.zeros(num_step+1) #Initialize loss_hist
    #TODO
    loss_hist[0] = compute_square_loss(X, y, theta) + lambda_reg * np.matmul(theta.transpose(), theta) # initial loss
    theta_hist[0] = theta
    for i in range(num_step):
        grad = compute_regularized_square_loss_gradient(X, y, theta, lambda_reg)
        theta = theta - alpha * grad
        theta_hist[i + 1] = theta
        loss = compute_square_loss(X, y, theta) + lambda_reg * np.matmul(theta.transpose(), theta)
        loss_hist[i + 1] = loss
    return theta_hist,loss_hist


#######################################
### Stochastic gradient descent
def stochastic_grad_descent(X, y, alpha=0.01, lambda_reg=10**-2, num_epoch=1000,C=0.1):
    """
    In this question you will implement stochastic gradient descent with regularization term

    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        alpha - string or float, step size in gradient descent
                NOTE: In SGD, it's not a good idea to use a fixed step size. Usually it's set to 1/sqrt(t) or 1/t
                if alpha is a float, then the step size in every step is the float.
                if alpha == "1/sqrt(t)", alpha = 1/sqrt(t).
                if alpha == "1/t", alpha = 1/t.
                t = num steps = (num epochs) x (num instances)
        lambda_reg - the regularization coefficient
        num_epoch - number of epochs to go through the whole training set

    Returns:
        theta_hist - the history of parameter vector, 3D numpy array of size (num_epoch, num_instances, num_features)
                     for instance, theta in epoch 0 should be theta_hist[0], theta in epoch (num_epoch) is theta_hist[-1]
        loss hist - the history of loss function vector, 2D numpy array of size (num_epoch, num_instances)
    """
    num_instances, num_features = X.shape[0], X.shape[1]
    theta = np.ones(num_features) #Initialize theta

    theta_hist = np.zeros((num_epoch, num_instances, num_features)) #Initialize theta_hist
    loss_hist = np.zeros((num_epoch, num_instances)) #Initialize loss_hist
    #TODO
    loss_hist[0] = skeleton_code.compute_square_loss(X, y, theta_hist[0][0]) #initial loss
    t = 0
    for i in range(num_epoch):
        idx = np.arange(num_instances)
        np.random.shuffle(idx)
        X = X[idx]
        y = y[idx]
        for j in range(num_instances):
            t = t+1
            if alpha =='1/sqrt(t)':
                step_size = C/(t**(1/2))
            elif alpha == '1/t':
                step_size = C/t
            else:
                step_size = alpha
            factor = np.matmul(X[j],theta) -  y[j]  #h(xi) - y(i)
            theta = theta - 2 * step_size * (factor * X[j].transpose() + lambda_reg * theta)
            theta_hist[i][j] = theta   
            loss_hist[i][j] = skeleton_code.compute_square_loss(X,y,theta) + lambda_reg * np.matmul(theta.transpose(), theta)
            
    return theta_hist, loss_hist

def main():
    #Loading the dataset
    print('loading the dataset')

    df = pd.read_csv('data.csv', delimiter=',')
    X = df.values[:,:-1]
    y = df.values[:,-1]

    print('Split into Train and Test')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =100, random_state=10)

    print("Scaling all to [0, 1]")
    X_train, X_test = feature_normalization(X_train, X_test)
    X_train = np.hstack((X_train, np.ones((X_train.shape[0], 1))))  # Add bias term
    X_test = np.hstack((X_test, np.ones((X_test.shape[0], 1))))  # Add bias term
    # TODO
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    main()
