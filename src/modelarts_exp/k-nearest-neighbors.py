"""
K近邻算法（K-Nearest Neighbors, KNN）分类器示例
使用sklearn库实现KNN分类，对二维数据点进行分类
"""

# 导入必要的库
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# 训练数据：6个二维数据点
X = np.array([
    [1, 1],      # 类别A
    [1, 1.5],    # 类别A
    [2, 2.5],    # 类别B
    [2.5, 3],    # 类别B
    [1.5, 1],    # 类别A
    [3, 2.5]     # 类别B
])
y = ['A', 'A', 'B', 'B', 'A', 'B']

# KNeighborsClassifier参数说明：
# - n_neighbors (k值): 指定使用最近的几个邻居进行投票，默认为5
# - algorithm: 指定用于计算最近邻的算法
#   * 'ball_tree': 球树算法，适用于高维数据
#   * 'kd_tree': KD树算法，适用于低维数据
#   * 'brute': 暴力搜索，适用于小数据集
#   * 'auto': 自动选择最合适的算法
model = KNeighborsClassifier(n_neighbors=4, algorithm='ball_tree')

# fit()函数说明：
# - 这是训练函数，但实际上KNN是"懒惰学习"算法
# - fit()只是存储训练数据，不进行实际计算
# - 真正的计算发生在预测阶段
model.fit(X, y)

if __name__ == "__main__":
    # 预测新样本
    new_sample = [[1.75, 1.75]]

    # 预测类别标签
    prediction = model.predict(new_sample)
    print(f"预测类别: {prediction[0]}")

    # 预测概率分布
    probabilities = model.predict_proba(new_sample)
    print(f"\npredict_proba()返回结果: {probabilities}")
    print(f"返回数组形状: {probabilities.shape}")
    print(f"probabilities[0][0] = {probabilities[0][0]:.3f}")
    print(f"probabilities[0][1] = {probabilities[0][1]:.3f}")

    # 获取类别标签的顺序
    class_labels = model.classes_
    print(f"\n类别标签顺序: {class_labels}")

    # 计算模型在训练数据上的准确率
    accuracy = model.score(X, y)
    print(f"模型准确率: {accuracy:.3f}")

"""
predict_proba()返回数组说明：
- 返回二维数组，形状为 (样本数, 类别数)
- 列的顺序对应 model.classes_ 中的类别顺序
- probabilities[i][j] 表示第i个样本属于第j个类别的概率
- 所有类别的概率和为1
"""
