from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
wine = datasets.load_wine()
x_train,x_test,y_train,y_test=train_test_split(wine.data,wine.target,test_size=0.3,random_state=321)
def Pca(data,k):
    data = data.T
    data_mean = np.mean(data, axis=1)
    data_mean = data_mean.reshape((len(data.T[0]), 1))
    data = data - data_mean
    C=np.dot(data, data.T) / len(data[1])
    eigen_vals,eigen_vecs=np.linalg.eig(C)
    #将特征值按照从大到小的顺序排序，选择其中最大的k个，
    # 然后将其对应的k个特征向量分别作为列向量组成特征向量矩阵
    eigValInd = np.argsort(eigen_vals)
    eigValInd = eigValInd[:-(k+1):-1]
    redEigVects = eigen_vecs[:,eigValInd]
    # 将样本点投影到选取的特征向量上
    lowDataMat=np.dot(data.T,redEigVects)
    return lowDataMat
lowDDataMat=Pca(x_train, 3)
print(lowDDataMat)