from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
import random,operator
iris = datasets.load_iris()
x_train,x_test,y_train,y_test=train_test_split(iris.data,iris.target,test_size=0.3,random_state=321)
trainSet=x_train.tolist()
y_train=y_train.tolist()
labels=iris.feature_names
for item1,item2 in zip(trainSet,y_train):
        item1.append(item2)
#y_train=np.reshape((1,len(y_train)))
# trainSet=np.array(trainSet)
# y_train=np.array(y_train)
def calcShang(dataSet):
    lenDataSet=len(dataSet)
    p={}
    H=0.0
    for data in dataSet:
        currentLabel=data[-1]  #获取类别标签
        if currentLabel not in p.keys():  #若字典中不存在该类别标签，即创建
            p[currentLabel]=0
        p[currentLabel]+=1    #递增类别标签的值
    for key in p:
        px=float(p[key])/float(lenDataSet)  #计算某个标签的概率
        H-=px*np.math.log(px,2)             #计算信息熵,用np.log报错了
    return H
#潜在分裂点，从第一个潜在分裂点开始，
#分裂D并计算两个集合的期望信息，
#具有最小期望信息的点称为这个属性的最佳分裂点，
#其信息期望作为此属性的信息期望。
# def findBound(dataset,axis):
#         return []
def spiltData(dataSet,axis,value,bound):    #dataSet为要划分的数据集,axis为给定的特征，value为给定特征的具体值,bound为分界点的值
    subDataSet=[]
    if value==">="+str(bound):
        for data in dataSet:
            subData=[]
            if data[axis]>=bound:
                subData=data[:axis]  #取出data中第0到axis-1个数进subData;
                subData.extend(data[axis+1:])  #取出data中第axis+1到最后一个数进subData
                subDataSet.append(subData)
    else:
        for data in dataSet:
            subData = []
            if data[axis] < bound:
                 subData = data[:axis]  # 取出data中第0到axis-1个数进subData;
                 subData.extend(data[axis + 1:])  # 取出data中第axis+1到最后一个数进subData
                 subDataSet.append(subData)
    return subDataSet

def chooseBestFeature(dataSet):
    lenFeature=len(dataSet[0])-1    #计算特征维度时要把类别标签那一列去掉
    shanInit=calcShang(dataSet)      #计算原始数据集的信息熵
    feature=[]
    inValue=0.0
    bestFeature=0
    for i in range(lenFeature):
        shanCarry=0.0
        #对于离散属性值，feature=[example[i] for example in dataSet]  #提取第i个特征的所有数据
        #对于连续属性值，得到第i个特征所有的潜在分裂点，如'1.2'、'2.2'
        # feature=findBound(dataSet, i)
        feature = [example[i] for example in dataSet]
        bound=np.mean(feature)
        bound=round(bound, 2)
        feature = [">=" + str(bound), "<" + str(bound)]
        for feat in feature:
            subData=spiltData(dataSet,i,feat,bound)  #先对数据集按照分类值分类
            prob=float(len(subData))/float(len(dataSet))
            shanCarry+=prob*calcShang(subData)  #计算第i个特征的信息熵
        outValue=shanInit-shanCarry  #原始数据信息熵与循环中的信息熵的差
        if (outValue>inValue):
            inValue=outValue  #将信息熵与原始熵相减后的值赋给inValue，方便下一个循环的信息熵差值与其比较
            bestFeature=i
    return bestFeature
def majorityCnt(classList):    #按分类后类别数量排序
    classCount={}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote]=0
        classCount[vote]+=1
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

def createTree(dataSet,labels):
    classList=[item[-1] for item in dataSet]  # 类别：0,1,2
    if classList.count(classList[0])==len(classList):
        return classList[0]
    if len(dataSet[0])==1:
        return majorityCnt(classList)
    bestFeat=chooseBestFeature(dataSet) #选择最优特征
    bestFeatLabel=labels[bestFeat]
    myTree={bestFeatLabel:{}} #分类结果以字典形式保存
    del(labels[bestFeat])
    featValues=[example[bestFeat] for example in dataSet]
    bound = np.mean(featValues)
    bound = round(bound, 2)
    uniqueVals=[">="+str(bound),"<"+str(bound)]
    for value in uniqueVals:
        subLabels = labels[:]   #[:]的用法
        myTree[bestFeatLabel][value]=createTree(spiltData
                                                (dataSet,bestFeat,value,bound),subLabels)
    return myTree
decisiontree=createTree(trainSet, labels)
print(decisiontree)