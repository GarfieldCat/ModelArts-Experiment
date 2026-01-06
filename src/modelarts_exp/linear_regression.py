# 一元线性回归的实现

#导入matplotlib库，主要用于可视化
import matplotlib.pyplot as plt
import numpy as np

# 构造用于训练的数据集
x_train = [4,8,5,10,12]
y_train = [20,50,30,70,60]

# 绘制散点图，显示训练数据点
def draw(x_train,y_train):
    plt.scatter(x_train, y_train)

# 定义函数求得斜率w和截距b
# 使用最小二乘法对斜率和截距求导并使得导数值等于0求解出斜率和截距
def fit(x_train,y_train):
    size = len(x_train)
    numerator = 0 #初始化分子
    denominator = 0#初始化分母
    for i in range(size):
        numerator += (x_train[i]-np.mean(x_train))*(y_train[i]-np.mean(y_train))
        denominator += (x_train[i]-np.mean(x_train))**2
    w = numerator/denominator
    # 截距公式推导自线性方程 y = wx + b：
    # - 两边取均值：mean(y) = w * mean(x) + b
    # - 解得：b = mean(y) - w * mean(x)
    b = np.mean(y_train)-w*np.mean(x_train)
    return w,b

#根据斜率w和截距b，输入x计算输出值
def predict(x,w,b):
    #预测模型
    y = w*x+b
    return y

# 根据W,B画图
def fit_line(w,b):
    # np.linspace() 参数说明：
    # - start = 4: 起始点，从x = 4 开始
    # - stop = 15: 结束点，到x = 15 结束
    # - num = 9: 生成9个点
    # 功能：创建一个包含9个点的等差数列：
    # [4.0, 5.375, 6.75, 8.125, 9.5, 10.875, 12.25, 13.625, 15.0]
    x = np.linspace(4,15,9)  #linspace 创建等差数列的函数
    y = w*x+b
    # - 在x - y坐标系中绘制一条连续的直线
    # - 连接所有计算出的(x, y)点
    # - 默认绘制蓝色实线
    plt.plot(x,y)
    plt.show()

if __name__ =="__main__":
    draw(x_train,y_train)
    w,b = fit(x_train,y_train)
    print(w,b) #输出斜率和截距

    fit_line(w,b) #绘制预测函数图像