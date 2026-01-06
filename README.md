# ModelArts-Experiment

ModelArts实验。人工智能工作级开发者认证的配套上机练习。

## 算法列表

- 线性回归算法：**linear_regression**
- 逻辑回归算法: **logistic_regression**
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

