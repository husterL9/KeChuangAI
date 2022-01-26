from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
iris = datasets.load_iris()
x_train,x_test,y_train,y_test=train_test_split(iris.data,iris.target,test_size=0.2,random_state=321)
def disR(a,b):
    a=a.reshape((1,4))#语法疑惑
    b=b.reshape((1,4))
    dis=a-b
    dis=dis**2
    res=np.sum(dis)
    return res**0.5
K=22
def knn(datax):
    res = [{ "target": target,"distance": disR(data,datax)}
           for data, target in zip(x_train, y_train)]
    #升序排序
    res=sorted(res,key=lambda item:item['distance'])
    #取前K个
    res2=res[0:K]
    #总距离
    sum=0
    for r in res2:
        sum+=r['distance']
    #加权平均
    result = {'0':0,'1':0,'2':0}
    for r in res2:
        result[repr(r['target'])]= 1-r['distance']/sum#为什么不加repr不行
    if result['0']>result['1']:
        if result['0']>result['2']:
            return '0'
        else:
            return '2'
    elif result['2']>result['1']:
        if result['0']>result['2']:
            return '0'
        else:
            return '2'
    else:return '1'

#测试
corret=0
for data, target in zip(x_test, y_test):
    result=repr(target)
    result2=knn(data)
    if result==result2:
        corret+=1
print(corret)
print(len(y_test))
print("准确率：{:.2f}%".format(100*corret/len(y_test)))



