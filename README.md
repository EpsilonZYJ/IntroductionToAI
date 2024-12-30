# Introduction to AI
## 环境配置
- 环境配置文件均在requirement目录下
- MacOS使用requirements_Mac.txt配置环境
- Windows使用requirements_Windows.txt配置环境
- 配置方法：命令行输入 
```shell
conda create --name <env> --file <this file>
```
- CUDA版本：>=12.4
## 目录构成
- **当前目录下：**
  - MobilePrice.ipynb、Titanic.ipynb分别为两个不同数据集的数据训练与可视化等
  - Titanic_predict.py为Titanic数据集中，利用已训练好的模型参数文件进行预测
- **Answer：** 存有保存的预测数据（仅Titanic数据集）
- **Data：** 两个数据集的训练集以及测试集
- **Model：** 模型参数以及保存的MinMax策略参数文件
- **Picture：** 模型训练以及数据可视化过程中绘制的图片
- **requirement：** 环境配置文件