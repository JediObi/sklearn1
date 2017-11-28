import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from drawline import plot_decision_regions


# 使用scikit集成的数据集
iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target

# 分割为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# 标准化数据，均值为0， 方差为1
# 使用sl的StandardScaler
sc = StandardScaler()
# 使用训练集计算均值和标准差，作为训练集和测试集的统一处理参数
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# 使用logistic
lr = LogisticRegression(C = 1000.0, random_state=0)
lr.fit(X_train_std, y_train)

# 合并标准化的特征集，原始样本被拆分，然后各自处理，所以此处合并
X_combined_std = np.vstack((X_train_std, X_test_std))
# 合并标签集，因为原始标签机被拆分过，所以要按特征集的顺序合并
y_combined = np.hstack((y_train, y_test))

plot_decision_regions(X_combined_std, y_combined, classifier=lr, test_idx=range(105, 150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.show()




