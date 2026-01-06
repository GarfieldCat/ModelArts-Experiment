import numpy as np
import os
#从sklearn导入LogisticRegression方法
from sklearn.linear_model import LogisticRegression
#导入划分训练集和测试集的方法
from sklearn.model_selection import train_test_split

# 获取项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_data(filename):
    """加载数据集"""
    filepath = os.path.join(PROJECT_ROOT, "data/logistic_regression", filename)
    return np.loadtxt(filepath, delimiter=",")


def train_and_evaluate(data, test_size=0.3):
    """
    训练并评估模型

    参数:
        data: 数据集
        test_size: 测试集占比，默认0.3

    提示:
        若需结果可复现，可在 train_test_split 中设置 random_state 参数，例如:
        train_test_split(X, y, test_size=test_size, random_state=42)
    """
    X, y = data[:, 0:2], data[:, 2]
    #划分训练集和测试集，测试集占数据集的30%，训练集占数据集的70%
    #train_test_split(x,y,test_size=0.3)     #参数test_size测试集占比; x:数据集; y:数据集的目标值
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=test_size)

    # 创建一个逻辑回归模型对象（此时还未训练）
    model = LogisticRegression()
    #通过训练集得到训练后的模型
    model.fit(train_x, train_y)

    #测试模型 对新数据做预测
    pred_y = model.predict(test_x)
    # 输出判断预测是否与真实值相等
    print(pred_y == test_y)

    return model, model.score(test_x, test_y)


if __name__ == "__main__":
    data = load_data("dataset_01.txt")
    print(data)
    model, accuracy = train_and_evaluate(data)
    print(f"模型准确率: {accuracy:.4f}")
