# Create by Yujie Zhou

from joblib import load
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import numpy as np
from torch.autograd import Variable
from torch.nn import functional as F

def EncodeData(x: pd.DataFrame) -> list:
    """
    对特征向量进行编码
    将pandas的DataFrame结构转换为list
    同时去除无关列
    :param x: pandas.DataFrame
    :return: 特征向量
    """
    input_data = [[x["Pclass"][i], x["Sex"][i], x["Age"][i], x["SibSp"][i], x["Parch"][i], x["Fare"][i], x["Embarked"][i]] for i in range(len(x))]
    
    return input_data

class PreEncodeData:
    def __init__(self) -> pd.DataFrame:
        """
        读取MinMax相关参数的数据进行初始化
        :return: pandas.DataFrame 编码后的数据
        """
        self.MinMax = MinMaxScalerStrategy()

    
    def PreEncode(self, x: pd.DataFrame) -> pd.DataFrame:
        """
        对再次输入数据进行预编码
        如将测试集数据进行与训练集相同的编码方式

        :param x: pandas.DataFrame 类型数据，为读取的数据
        :return: pandas.DataFrame 编码后的数据
        """
        for i in range(len(x["Sex"])):
            if x["Sex"][i] == 'male':
                x.loc[i, "Sex"] = 1
            elif x["Sex"][i] == 'female':
                x.loc[i, "Sex"] = 0
        
        for i in range(len(x["Embarked"])):
            if x["Embarked"][i] == 'C':
                x.loc[i, "Embarked"] = 0
            elif x["Embarked"][i] == 'Q':
                x.loc[i, "Embarked"] = 1
            elif x["Embarked"][i] == 'S':
                x.loc[i, "Embarked"] = 2
        x = x.drop(columns=["Name", "Ticket", "Cabin"])
        x = self.MinMax.transform(x)
        return x

def EncodeTrainData(x: pd.DataFrame) -> tuple[list, list]:
    """
    对测试集进行编码处理
    将pandas的DataFrame结构转换为list
    同时去除无关列
    :param x: pandas.DataFrame
    :return: 特征向量，标签向量
    """
    input_data = EncodeData(x)
    label = [x["Survived"][i] for i in range(len(x))]

    return input_data, label

class MinMaxScalerStrategy:
    def __init__(self):
        data = pd.read_csv(
            filepath_or_buffer="Model/Titanic_MinMax.csv",
            names=["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"],
            skiprows=1
        )
        self.min = {
            "Pclass": data["Pclass"][1],
            "Sex": data["Sex"][1],
            "Age": data["Age"][1],
            "SibSp": data["SibSp"][1],
            "Parch": data["Parch"][1],
            "Fare": data["Fare"][1],
            "Embarked": data["Embarked"][1]
        }
        self.max = {
            "Pclass": data["Pclass"][0],
            "Sex": data["Sex"][0],
            "Age": data["Age"][0],
            "SibSp": data["SibSp"][0],
            "Parch": data["Parch"][0],
            "Fare": data["Fare"][0],
            "Embarked": data["Embarked"][0]
        }


    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        colunm_names = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
        for name in colunm_names:
            for i in range(len(data[name])):
                data.loc[i, name] = (data[name][i] - self.min[name]) / (self.max[name] - self.min[name])
        return data

class MachineLearningModel:
    def __init__(self, filepath: str):
        self.model = load(filepath)
    
    def getModel(self):
        return self.model
    
    def predict(self, x):
        return self.model.predict(x)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(8, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 2)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

if __name__ == '__main__':
    test_DataFrame = pd.read_csv(filepath_or_buffer="Data/Titanic/test.csv", 
                    names=["PassengerId", "Pclass", "Name", "Sex", "Age", 
                    "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked"], skiprows=1)
    test_AnswerDataFrame = pd.read_csv(filepath_or_buffer="Data/Titanic/gender_submission.csv",
                    names=["PassengerId", "Survived"], skiprows=1)

    imputer = KNNImputer(n_neighbors=5, weights="distance", copy=False)

    test_DataFrame_raw = test_DataFrame.copy()
    '''
    数据预处理
    '''

    # 进行编码初始化
    EncodeModel = PreEncodeData()

    # 测试集编码
    print("test_DataFrame:")
    test_DataFrame = EncodeModel.PreEncode(test_DataFrame)
    print(test_DataFrame)

    # 缺失值填充
    print("test_DataFrame_filled:")
    test_DataFrame_filled = pd.DataFrame(imputer.fit_transform(test_DataFrame), columns=test_DataFrame.columns)
    print(test_DataFrame_filled)

    print("test_dataset:")
    test_dataset = EncodeData(test_DataFrame_filled)
    # print(test_dataset)
    test_label = [test_AnswerDataFrame["Survived"][i] for i in range(len(test_AnswerDataFrame))]
    
    '''
    模型预测
    '''

    '''
    SVM
    '''

    SVM_model = MachineLearningModel("Model/Titanic_SVM_model.joblib")

    label_pred = SVM_model.predict(test_dataset)

    SVM_prediction = pd.DataFrame({
        "PassengerId:": test_DataFrame_raw["PassengerId"],
        "Survived": [int(item) for item in label_pred]
    })
    SVM_prediction.to_csv("Answer/Titanic_SVM_submission.csv", index=False)

    print("-------------------------------------------------------------")
    print("SVM:")
    # 计算准确率
    accuracy = accuracy_score(test_label, label_pred)
    print(f"Accuracy: {accuracy:.2f}")

    # 打印分类报告
    print("Classification Report:")
    print(classification_report(test_label, label_pred))

    # 打印混淆矩阵
    print("Confusion Matrix:")
    print(confusion_matrix(test_label, label_pred))
    print("-------------------------------------------------------------")
    
    '''
    RandomForest
    '''

    RF_model = MachineLearningModel("Model/Titanic_RF_model.joblib")

    label_pred = RF_model.predict(test_dataset)

    RF_prediction = pd.DataFrame({
        "PassengerId:": test_DataFrame_raw["PassengerId"],
        "Survived": [int(item) for item in label_pred]
    })
    RF_prediction.to_csv("Answer/Titanic_RF_submission.csv", index=False)

    print("-------------------------------------------------------------")
    print("RandomForest:")
    # 计算准确率
    accuracy = accuracy_score(test_label, label_pred)
    print(f"Accuracy: {accuracy:.2f}")

    # 打印分类报告
    print("Classification Report:")
    print(classification_report(test_label, label_pred))

    # 打印混淆矩阵
    print("Confusion Matrix:")
    print(confusion_matrix(test_label, label_pred))
    print("-------------------------------------------------------------")

    '''
    MLP
    '''

    MLP_model = MachineLearningModel("Model/Titanic_MLP_model.joblib")

    label_pred = MLP_model.predict(test_dataset)

    MLP_prediction = pd.DataFrame({
        "PassengerId:": test_DataFrame_raw["PassengerId"],
        "Survived": [int(item) for item in label_pred] 
    })
    MLP_prediction.to_csv("Answer/Titanic_MLP_submission.csv", index=False)

    print("-------------------------------------------------------------")
    print("MLP:")
    # 计算准确率
    accuracy = accuracy_score(test_label, label_pred)
    print(f"Accuracy: {accuracy:.2f}")

    # 打印分类报告
    print("Classification Report:")
    print(classification_report(test_label, label_pred))

    # 打印混淆矩阵
    print("Confusion Matrix:")
    print(confusion_matrix(test_label, label_pred))

    print("-------------------------------------------------------------")

    '''
    使用线性神经网络模型
    '''

    # 准备数据集
    df_test  = pd.read_csv('./Data/Titanic/test.csv')
    df_sub   = pd.read_csv('./Data/Titanic/gender_submission.csv')

    df_test.drop( ['Name','Ticket','Cabin'],axis=1,inplace=True)

    sex     = pd.get_dummies(df_test['Sex'],drop_first=True)
    embark  = pd.get_dummies(df_test['Embarked'],drop_first=True)
    df_test = pd.concat([df_test,sex,embark],axis=1)

    df_test.drop(['Sex','Embarked'],axis=1,inplace=True)

    df_test.fillna(df_test.mean(),inplace=True)
    
    Scaler2 = StandardScaler()

    test_columns  = df_test.columns
    
    df_test  = pd.DataFrame(Scaler2.fit_transform(df_test))
    
    df_test.columns  = test_columns

    # 加载模型
    model = Net()
    model.load_state_dict(torch.load("Model/Titanic_Linear_model.pt"))

    print("-------------------------------------------------------------")
    print("Linear Neural Network:")
    # 预测
    X_test     = df_test.iloc[:,1:].values
    X_test_var = Variable(torch.FloatTensor(X_test), requires_grad=False) 
    with torch.no_grad():
        test_result = model(X_test_var)
    values, labels = torch.max(test_result, 1)
    survived = labels.data.numpy()

    # 评估正确率
    y_pred_df = pd.read_csv('Answer/Titanic_submission.csv', names=["PassengerId", "Survived"], skiprows=1)
    y_test_df = pd.read_csv('Data/Titanic/gender_submission.csv', names=["PassengerId", "Survived"], skiprows=1)

    y_pred = [y_pred_df["Survived"][i] for i in range(len(y_pred_df))]
    y_test = [y_test_df["Survived"][i] for i in range(len(y_test_df))]
    
    submission = pd.DataFrame({'PassengerId': df_sub['PassengerId'], 'Survived': survived})
    submission.to_csv('Answer/Titanic_submission.csv', index=False)

    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")

    # 打印分类报告
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # 打印混淆矩阵
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("-------------------------------------------------------------")