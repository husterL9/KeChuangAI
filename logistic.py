from sklearn import datasets
from sklearn.model_selection import train_test_split
# import numpy as np
from numpy import *
iris = datasets.load_iris()
# print(dir(iris))
# print("num data is: {}, num labels is: {}"
#        .format(len(iris.data), len(iris.target)))
x_train,x_test,y_train,y_test=train_test_split(iris.data,iris.target,test_size=0.2,random_state=321)
x_train=x_train.T
x_test=x_test.T
y_train=y_train.reshape((1,len(y_train)))
y_test=y_test.reshape((1,len(y_test)))
def sigmoid(Z):
	return 1.0 / (1 + exp(-Z))
def initialize_with_zeros(dim):
    W = zeros(shape = (dim,1))
    b = 0
    assert(W.shape == (dim, 1))
    assert(isinstance(b, float) or isinstance(b, int))

    return (W , b)
def propagate(W, b, X, Y):
    m=X.shape[1]
    Z = dot(W.T, x_train) + b
    A=sigmoid(Z)
    cost = (- 1 / m) * sum(Y * log(A) + (1 - Y) * (log(1 - A)))
    dZ=A-Y
    dW=1/m*dot(x_train,dZ.T)
    db=1/m*sum(dZ)
    grads = {
                "dW": dW,
                "db": db
             }
    return grads
def optimize(W , b , X , Y , num_iterations , learning_rate ):
    for i in range(num_iterations):
        grads= propagate(W, b, X, Y)
        dW = grads["dW"]
        db = grads["db"]
        W= W - learning_rate * dW
        b = b - learning_rate * db
        params = {
            "W": W,
            "b": b}
        grads = {
            "dW": dW,
            "db": db}
        return (params, grads)
def predict(W , b , X ):
    m = X.shape[1]
    Y_prediction = zeros((1, m))
    A = sigmoid(dot(W.T, X) + b)
    for i in range(A.shape[1]):
        Y_prediction[0, i] = 1 if A[0, i] > 0.5 else 0
    return  Y_prediction
def model(k,x_train , y_train , x_test , y_test , num_iterations = 2000 , learning_rate = 0.5 ):
    #转变为二分分类（k=0,1,2）
    if k==0:
        for j in range(y_train.shape[1]):
            if y_train[0][j] == 0:
               y_train[0][j]=1
            else:
               y_train[0][j]=0
        for j in range(y_test.shape[1]):
            if y_test[0][j] == 0:
               y_test[0][j]=1
            else:
               y_test[0][j]=0
    elif k==1:
        for j in range(y_train.shape[1]):
            if y_train[0][j] == 1:
               y_train[0][j]=1
            else:
               y_train[0][j]=0
        for j in range(y_test.shape[1]):
            if y_test[0][j] == 1:
               y_test[0][j]=1
            else:
               y_test[0][j]=0
    else:
        for j in range(y_train.shape[1]):
            if y_train[0][j] == 2:
               y_train[0][j]=1
            else:
               y_train[0][j]=0
        for j in range(y_test.shape[1]):
            if y_test[0][j] == 2:
               y_test[0][j]=1
            else:
               y_test[0][j]=0
    W, b = initialize_with_zeros(x_train.shape[0])
    parameters, grads = optimize(W, b, x_train, y_train, num_iterations, learning_rate)
    W, b = parameters["W"], parameters["b"]
    Y_prediction_test = predict(W, b,x_test)
    Y_prediction_train = predict(W, b,x_train)
    print("训练集准确性：", format(100 - mean(abs(Y_prediction_train - y_train)) * 100), "%")
    print("测试集准确性：", format(100 - mean(abs(Y_prediction_test - y_test)) * 100), "%")

    d = {
        "Y_prediction_test": Y_prediction_test,
        "Y_prediciton_train": Y_prediction_train,
        "W": W,
        "b": b,
        "learning_rate": learning_rate,
        "num_iterations": num_iterations
    }
    return d
d = model(2,x_train, y_train, x_test, y_test, num_iterations = 10000, learning_rate = 0.005)