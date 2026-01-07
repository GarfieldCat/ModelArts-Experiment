import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def loadDataSet(fileName):
    """
    Args:
        fileName 文件名
    Returns:
        dataMat  数据矩阵
        labelMat 类标签
    """
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat, labelMat


def main():
    X, Y = loadDataSet(os.path.join(PROJECT_ROOT, 'data/support-vector-machine/testSet.txt'))
    X = np.asarray(X)

    print("X=", X[:5])
    print("Y=", Y[:5])

    # 训练模型
    # C: 正则化参数，值越大对误分类惩罚越大，边界越窄；值越小容忍更多误分类，边界越宽
    # kernel: 核函数类型，'linear'表示线性核，适用于线性可分数据
    # gamma: 核函数系数，仅对'rbf'/'poly'/'sigmoid'核有效，linear核会忽略此参数
    clf = svm.SVC(C=5, kernel='linear', gamma=10)
    # 训练 SVM 模型
    clf.fit(X, Y)

    # 获取训练后的超平面权重向量 w = [w1, w2]
    # 超平面方程为： w1*x + w2*y + b = 0
    # 转换为直线方程 y = ax + c
    # y = (-w1/w2)*x - b/w2
    # coef_[0] 取第一个分类器的系数（二分类只有一个）
    w = clf.coef_[0]
    # 斜率
    a = -w[0] / w[1]
    # 从-2到10，顺序间隔采样50个样本，默认是num=50
    xx = np.linspace(-2, 10)  # , num=50)
    # 二维的直线方程
    yy = a * xx - (clf.intercept_[0]) / w[1]
    print("yy=", yy)

    # 通过支持向量绘制分割超平面
    print("support_vectors_=", clf.support_vectors_)
    b = clf.support_vectors_[0]
    yy_down = a * xx + (b[1] - a * b[0])
    b = clf.support_vectors_[-1]
    yy_up = a * xx + (b[1] - a * b[0])

    # 画出直线，散点以及临界点的支持向量平面
    plt.plot(xx, yy, 'k-')
    plt.plot(xx, yy_down, 'k--')
    plt.plot(xx, yy_up, 'k--')

    # 绘制支持向量点
    plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=80, facecolors='none', edgecolors='red', linewidths=2, zorder=10)
    # 绘制所有数据点
    plt.scatter(X[:, 0].flat, X[:, 1].flat, c=Y, cmap=plt.cm.Paired, zorder=5)
    plt.axis('tight')
    plt.show()

if __name__ == "__main__":
    main()