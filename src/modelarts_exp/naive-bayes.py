import numpy as np
from sklearn.naive_bayes import BernoulliNB


def main():
    """
    朴素贝叶斯分类器示例
    演示模型训练和预测功能
    """
    print("朴素贝叶斯分类器示例")
    print("=" * 40)

    # 1. 生成示例数据
    print("\n1. 生成示例数据...")
    X = np.random.randint(2, size=(6, 100))  # 6个样本，100个二进制特征
    y = np.array([1, 2, 3, 4, 4, 5])        # 对应的类别标签
    print(f"   特征矩阵形状: {X.shape}")
    print(f"   类别标签: {y}")

    # 2. 训练朴素贝叶斯分类器
    print("\n2. 训练模型...")
    clf = BernoulliNB()
    clf.fit(X, y)
    print("   ✓ 训练完成！")

    # 3. 使用训练好的模型进行预测
    print("\n3. 预测样本...")
    # 取第3个样本（索引为2）进行预测
    test_sample = X[2:3]  # 形状: (1, 100)
    prediction = clf.predict(test_sample)
    prediction_proba = clf.predict_proba(test_sample)

    print(f"   测试样本索引: 2")
    print(f"   预测结果: {prediction[0]}")
    print(f"   实际标签: {y[2]}")
    print(f"   预测正确: {'✓' if prediction[0] == y[2] else '✗'}")

    # 4. 显示预测概率
    print(f"\n4. 各类别预测概率:")
    for i, class_label in enumerate(clf.classes_):
        prob = prediction_proba[0][i]
        print(f"   类别 {class_label}: {prob:.4f}")

    # 5. 模型信息
    print(f"\n5. 模型信息:")
    print(f"   学习到的类别: {clf.classes_}")
    print(f"   特征数量: {clf.n_features_in_}")


if __name__ == "__main__":
    main()