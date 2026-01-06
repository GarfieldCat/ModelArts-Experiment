#从sklearn 导入KNeighborsClassifier方法
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

X = np.array([[1,1],[1,1.5],[2,2.5],[2.5,3],[1.5,1],[3,2.5]])
y = ['A','A','B','B','A','B']

# n_neighbors int 型参数 knn算法中指定以最近的几个最近邻样本具有投票权，默认参数为5
# 'ball_tree':球树、'kd_tree':kd树、'brute':暴力搜索、'auto':自动根据数据的类型和结构选择合适的算法。
model = KNeighborsClassifier(n_neighbors=4, algorithm='ball_tree')

"""
fit 就是训练模型
fit()                     
训练函数，它是最主要的函数。接收参数只有1个，就是训练数据集，
每一行是一个样本，每一列是一个属性。它返回对象本身，即只是修
改对象内部属性，因此直接调用就可以了，后面用该对象的预测函数
取预测自然及用到了这个训练的结果。其实该函数并不是
KNeighborsClassifier这个类的方法，
而是它的父类SupervisedIntegerMixin继承下来的方法。
"""
model.fit(X, y)

#预测
print(model.predict([[1.75,1.75]])) #输出分类结果
print(model.predict_proba([[1.75,1.75]])) #返回预测属于某标签的概率
print(model.score(X,y)) #输出模型训练结果