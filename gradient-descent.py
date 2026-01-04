import numpy as np
import matplotlib.pyplot as plt

def generate_gradient(X, theta, y):
    """
    计算损失函数对参数theta的梯度。
    参数:
    X: 特征矩阵，形状为(m, n)，其中m是样本数，n是特征数。
    theta: 模型参数，形状为(n, 1)。
    y: 目标变量，形状为(m, 1)。
    返回:
    gradient: 梯度，形状为(n, 1)。
    """
    m = len(y)
    h = X.dot(theta)  # 计算预测值
    gradient = (1/m) * X.T.dot(h - y)  # 计算梯度
    return gradient


def gradient_descending(X, y, theta, alpha):
    """
    使用梯度下降算法优化模型参数theta。
    参数:
    X: 特征矩阵，形状为(m, n)。
    y: 目标变量，形状为(m, 1)。
    theta: 初始参数，形状为(n, 1)。
    alpha: 学习率。
    返回:
    theta: 优化后的参数，形状为(n, 1)。
    Jthetas: 记录损失函数值的列表，用于绘制损失函数变化趋势。
    """
    Jthetas = []  # 用于记录损失函数值
    m = len(y)
    index = 0

    # 计算初始梯度
    gradient = generate_gradient(X, theta, y)

    # 开始梯度下降迭代
    while not np.all(np.absolute(gradient) <= 1e-5):  # 当梯度绝对值小于等于1e-5时停止
        # 更新参数
        theta = theta - alpha * gradient

        # 计算新梯度
        gradient = generate_gradient(X, theta, y)

        # 计算损失函数
        h = X.dot(theta)
        Jtheta = (1 / (2 * m)) * (h - y).T.dot(h - y)

        # 每10次迭代记录一次损失函数值
        if (index + 1) % 10 == 0:
            Jthetas.append((index, Jtheta[0]))

        index += 1

    return theta, Jthetas

# 生成示例数据
np.random.seed(42)  # 设置随机种子，确保结果可重复
X = np.random.rand(100, 1) * 10  # 生成100个样本，特征范围在0到10之间
y = 2 * X + 1 + np.random.randn(100, 1) * 1  # 真实模型为y=2x+1，添加噪声

# 添加偏置项（截距）
X = np.hstack((np.ones((X.shape[0], 1)), X))  # 在特征矩阵中添加一列1，用于偏置项

# 初始化参数
theta = np.random.randn(2, 1)  # 初始参数，包括偏置项和权重

# 设置超参数
alpha = 0.01  # 学习率

theta, Jthetas = gradient_descending(X, y, theta, alpha)

# 提取损失函数值和对应的迭代次数
iterations = [item[0] for item in Jthetas]
loss_values = [item[1] for item in Jthetas]

# 绘制损失函数变化趋势
plt.figure(figsize=(10, 5))
plt.plot(iterations, loss_values, label='Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Loss vs. Iterations')
plt.legend()
plt.show()