# -*- coding:utf-8 -*-

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from drawline import plot_decision_regions
import matplotlib.pyplot as plt
import time

'''
这个类用用于展示调用scikit-learn封装的感知器
使用直线划分三个种类的iris(取两个特征)
'''

iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target
# 把数据及分为训练集(0.7)和测试集(0.3)
print(time.strftime("%Y-%m-%d",time.localtime(time.time()))+"  "+"y.shape:"+str(y.shape))
# print("a"+str(1))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
print(time.strftime("%Y-%m-%d",time.localtime(time.time()))+"  "+"y.train.shape:"+str(y_train.shape), "  "+"y_test.shape:"+str(y_test.shape))
a = range(1,3)

# print(X_train.shape, X_test.shape)

# 标准化数据
# 新建该对象
sc = StandardScaler()
# 计算样本的特征均值和标准差
sc.fit(X_train)
# 生成标准化样本--训练集
X_train_std = sc.transform(X_train)
# 使用训练集的标准化参数处理测试集
X_test_std = sc.transform(X_test)

# 实例化感知机
ppn = Perceptron(n_iter=40, eta0=0.1, random_state=0)
# 训练模型
ppn.fit(X_train_std, y_train)

# 使用测试集预测结果
y_perd = ppn.predict(X_test_std)
# 打印测试出错的个数
print('Missclassified samples: %d'%(y_test!=y_perd).sum())

# 合并标准化样本集。因为标准化过程单独生成了两个样本集
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

# 绘制测试结果
plot_decision_regions(X = X_combined_std, y = y_combined, classifier= ppn, test_idx=range(105,150))
plt.xlabel('petal length [standardized]')
plt.ylabel('sepal length [standardized')
plt.legend(loc = 'upper left')
plt.show()