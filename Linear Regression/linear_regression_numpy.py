import numpy as np

def linear_regression_noreg(X, y):
    """
  Compute the weight parameter given X and y. \beta = (X^TX)^{-1}X^TY
  Inputs:
  - X: A numpy array of shape (num_samples, D) containing features
  - y: A numpy array of shape (num_samples, ) containing labels
  Returns:
  - w: a numpy array of shape (D, )
  """
    w = np.matmul(np.linalg.inv(np.matmul(X.T, X)), np.matmul(X.T, y))
    return w

def regularized_linear_regression(X, y, lambd):
    """
    Compute the weight parameter given X, y and lambda.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing features
    - y: A numpy array of shape (num_samples, ) containing labels
    - lambd: a float number specifying the regularization parameter
    Returns:
    - w: a numpy array of shape (D, )
    """
    w = np.matmul(np.linalg.inv(np.matmul(X.T, X) + lambd * np.identity(X.shape[1])), np.matmul(X.T, y))
    return w