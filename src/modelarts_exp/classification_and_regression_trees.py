import os

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_data():
    """导入数据集"""
    data = pd.read_csv(
        os.path.join(PROJECT_ROOT, 'data/classification_and_regression_trees/titanic_train.csv'),
        sep=','
    )
    print(data.head())
    data.info()
    return data


def handle_missing_values(data):
    """处理缺失值"""
    # 计算各特征缺失总数
    total = data.isnull().sum().sort_values(ascending=False)
    # 计算各特征缺失比例
    #   | 方法      | 作用                              |
    #   |----------|---------------------------------|
    #   | .sum()   | 统计 True 的数量（因为 True=1, False=0） |
    #   | .count() | 统计非 NA 值的数量（布尔值都是有效值，所以等于总行数）   |
    percent = (data.isnull().sum() / data.isnull().count()).sort_values(ascending=False)
    # 把两个 Series 横向合并成一个 DataFrame
    miss_data = pd.concat([total, percent], axis=1, keys=['Miss_Total', 'Miss_Percent'])
    print(miss_data.head())

    # 删除'Cabin'
    del data['Cabin']
    # 采用中位数填充缺失值
    data['Age'] = data['Age'].fillna(data['Age'].median())
    # 众数（出现次数最多的值）填充缺失值
    data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])
    data.info()
    return data


def feature_engineering(data):
    """特征工程：对乘客的Title进行处理"""
    # 观察Name特征提取其中的Title称呼
    data['Title'] = data['Name'].str.split(",", expand=True)[1].str.split(".", expand=True)[0]
    # 将字符型变量做数值化处理
    label = LabelEncoder()
    data['Sex_Code'] = label.fit_transform(data['Sex'])
    data['Title_Code'] = label.fit_transform(data['Title'])
    data['Embarked'] = data['Embarked'].astype(str)
    data['Embarked_Code'] = label.fit_transform(data['Embarked'])
    # 考虑到PassengerId和Ticker为随机生成的变量，不作为影响目标变量的信息，因此特征选择时，将其去除
    features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_Code', 'Title_Code', 'Embarked_Code', 'Survived']
    data = data[features]
    print(data.head())
    return data


def split_data(data):
    """划分训练集和测试集"""
    X = data[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_Code', 'Title_Code', 'Embarked_Code']]
    y = data[['Survived']]
    # random_state为随机种子，确保每次划分的结果是相同的
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
    return X_train, X_test, y_train, y_test


def train_and_evaluate(X_train, X_test, y_train, y_test):
    """训练模型并评估"""
    dtc = DecisionTreeClassifier()
    dtc.fit(X_train, y_train)
    # 预测
    y_predict = dtc.predict(X_test)

#   混淆矩阵概念（以泰坦尼克生存预测为例）：
#   实际\预测    预测死亡    预测生存
#   实际死亡        80          15        → 假正率：15/(80+15)=15.8%
#   实际生存        12          73        → 假负率：12/(12+73)=14.1%
    # 模型评分：准确率，查全率，查准率，F1得分
    # 准确率 (Accuracy): (80+73)/(80+73+12+15) = 86.7%
    acc = accuracy_score(y_test, y_predict)
    # 召回率 (Recall): 73/(12+73) = 85.9%
    rec = recall_score(y_test, y_predict)
    # 精确率 (Precision): 73/(15+73) = 83.0%
    prec = precision_score(y_test, y_predict)
    # F1分数: 2×(0.859×0.830)/(0.859+0.830) = 84.4%
    f1 = f1_score(y_test, y_predict)
    print("DecisionTreeClassifier Results")
    print("Accuracy      :", acc)
    print("Recall        :", rec)
    print("Precision     :", prec)
    print("F1 Score      :", f1)
    return dtc
#   选择场景：
#   - 关注整体准确率 → 看 Accuracy
#   - 不漏掉任何一个生存者 → 提升 Recall
#   - 预测生存必须准确 → 提升 Precision
#   - 平衡 Precision 和 Recall → 看 F1


def grid_search(dtc, X_train, X_test, y_train, y_test):
    """网格搜索进行参数调优"""
    param = {'max_depth': [1, 3, 5, 7]}
    #   | 参数         | 说明       | 值                            |
    #   |------------|----------|------------------------------|
    #   | estimator  | 要优化的模型对象 | 决策树 DecisionTreeClassifier() |
    #   | param_grid | 要搜索的参数组合 | {'max_depth': [1, 3, 5, 7]}  |
    #   | cv         | 交叉验证折数   | 5（将训练集分成5份）                  |
    #   | scoring    | 评估指标     | 'f1'（用F1分数评估模型好坏）            |
    gsearch = GridSearchCV(estimator=dtc, param_grid=param, cv=5, scoring='f1')
    gsearch.fit(X=X_train, y=y_train)
    print("最优参数：{}".format(gsearch.best_params_))
    print("最优模型：{}".format(gsearch.best_estimator_))
    print("模型最高分：{:.3f}".format(gsearch.score(X_test, y_test)))
    return gsearch.best_params_


def predict_with_best_model(X_train, X_test, y_train, y_test, best_params):
    """选择最优模型进行预测"""
    dtc = DecisionTreeClassifier(**best_params)
    dtc.fit(X_train, y_train)
    y_predict = dtc.predict(X_test)
    # 打印预测结果
    print('===================预测值=======================')
    print(y_predict)
    # 打印真实值
    print('===================真实值=======================')
    print(np.array(y_test).tolist())


def main():
    # 加载数据
    data = load_data()
    # 处理缺失值
    data = handle_missing_values(data)
    # 特征工程
    data = feature_engineering(data)
    # 划分数据集
    X_train, X_test, y_train, y_test = split_data(data)
    # 训练并评估
    dtc = train_and_evaluate(X_train, X_test, y_train, y_test)
    # 网格搜索
    best_params = grid_search(dtc, X_train, X_test, y_train, y_test)
    # 最优模型预测
    predict_with_best_model(X_train, X_test, y_train, y_test, best_params)


if __name__ == '__main__':
    main()