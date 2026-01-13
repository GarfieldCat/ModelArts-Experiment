# ModelArts-Experiment

ModelArts实验。人工智能工作级开发者认证的配套上机练习。

## 算法列表

- 线性回归算法：**linear_regression**
- 逻辑回归算法: **logistic_regression**
- K近邻算法（简称 **KNN**）: **k-nearest-neighbors**
- 朴素贝叶斯算法: **naive-bayes** 
- 支持向量机（简称 **SVM**）: **support-vector-machine**
- 迭代二叉树3（简称 **ID3**）: Iterative Dichotomiser 
- 决策树C4.5算法: 
- 分类回归树（简称 **CART**）: **classification_and_regression_trees**
- 梯度下降算法：**gradient_descent**

## 操作指令

**启动虚拟环境**
```
conda activate ModelArts-Experiment
```

**生成包管理文件**
```
conda env export --no-builds | findstr /v "prefix" > environment.yml
```  

