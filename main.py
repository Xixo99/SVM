import numpy as np
import matplotlib.pyplot as plt

def loadDataSet():
    x_train = np.load('./data/s-svm/train_data.npy')
    y_train = np.load('./data/s-svm/train_target.npy')
    x_test = np.load('./data/s-svm/test_data.npy')
    y_test = np.load('./data/s-svm/test_target.npy')

    return x_train, y_train, x_test, y_test

def gradAscent(dataSet, label):
    dataMat = np.mat(dataSet)   #  (m,n)
    # print('dataMate', dataMat)
    m, n = np.shape(dataMat)
    # print(m, n)
    lableMat = np.mat(label).transpose()  ## (m,1)
    # print('lableMat', lableMat)

    alpha = 0.5
    C = 0.7      # 惩罚因子
    maxCycle = 100000

    weights = np.zeros(m)
    b = 0

    print('size:', dataMat.size, weights.size)
    for i in range(maxCycle):
        e = lableMat - dataMat[:,1] * (weights * dataMat[:,0] + b)       # 计算误差向量 (m, 1)
        # print('e: ', e)
        i = np.argmax(e)
        # print('i: ', i)
        weights = (1 - alpha) * weights + alpha * C * dataMat[i,0] * dataMat[i,1]
        b = b + alpha * C * dataMat[i,1]
        # print(weights, b)

    w = np.asarray(weights).squeeze()
    # print('--------w:', w[0])
    return w[0], b

def plotBestFit(dataSet, label, weights, b):
    print('label',label)
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []

    m = len(dataSet)
    n = len([dataSet[0]])

    for i in range(m):
        if (label[i] == -1):
            xcord1.append(dataSet[i][0])
            ycord1.append(dataSet[i][1])
        elif label[i] == 1:
            xcord2.append(dataSet[i][0])
            ycord2.append(dataSet[i][1])

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.scatter(xcord1, ycord1, s= 30, c = 'blue', marker = 's')    ## 两种点分开画
    ax.scatter(xcord2, ycord2, s= 30, c = 'red',)

    x = np.arange(-4.0, 4.0, 0.10)
    print("------weight:-----", weights, b)
    y = weights * x + b
    print('---y: ',y)
    ax.plot(x,y)                                         # 画拟合直线

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

def test(x_test, y_test, weights, b):
    count = 0
    total = len(x_test)
    for i in range(total):
        result = classifyVector(x_test[i], weights, b)
        if (abs(result - y_test[i]) < 0.000001):    # 浮点数判断相等要注意细节
            count += 1
    return count/total

def train():
    x_train, y_train, x_test, y_test = loadDataSet()
    # print(x_train, y_train)
    w, b = gradAscent(x_train, y_train)
    # print('w,b:', w , b)
    plotBestFit(x_train, y_train, w, b)
    # print(x_test, y_test)
    acc = test(x_test, y_test, w, b)
    print('------acc------\n')
    print('finnal acc in test:', acc)

def classifyVector(x_test, weights, b):
    x = x_test[0]
    y = x_test[1]
    pred = weights* x + b
    # 考虑点在线的上方还是下方返回预测结果
    if (pred > y):
        return 1.0
    else:
        return -1.0

if __name__ == '__main__':
    train()