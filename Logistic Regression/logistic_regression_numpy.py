import numpy as np
    
def sigmoid_activation(x):
    return 1/(1 + np.exp(-x))

def predict_prob(theta, X):
    return sigmoid_activation(np.dot(X, theta))

def predict(theta, X, threshold=0.5):
    return predict_prob(theta, X) >= threshold

def model_fit(X, y, lr, num_iter):
    # initialize weight
    
    intercept = np.ones((X.shape[0], 1))
    X = np.concatenate((intercept, X), axis=1)
    theta = np.zeros(X.shape[1])
    
    prev = theta

    for i in range(num_iter):
        z = np.dot(X, theta)
        h = sigmoid_activation(z)
        gradient = np.dot(X.T,(y - h))/ y.shape[0]
        theta += lr * gradient

        if i % 1000 == 0:
            y_pred = predict(theta, X)
            print(i, np.sum(y_pred==y)/y.shape[0])

    return theta
    