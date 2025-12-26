'''
插值和预测模型，如：Kriging Model
Author: Dysin
Time:   2024.07.02
'''
import sys

import torch
import numpy as np
import pandas as pd
from pyKriging.krige import kriging
from smt.surrogate_models import KRG
from sklearn.model_selection import train_test_split
from uq.error_analysis import ErrorBasic
from uq.model_test import ModelTest

class SurrogateModelBasic:
    def __init__(self, params_input, params_output):
        '''
        代理模型
        :param params_input:    输入参数
        :param params_output:   输出参数
        '''
        self.x = self.to_numpy(params_input)
        self.y = self.to_numpy(params_output)
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.x_tensor = self.to_torch(self.x)
            self.y_tensor = self.to_torch(self.y)
        else:
            self.device = torch.device('cpu')

    def to_numpy(self, data):
        """
        自动识别类型并转为 numpy.ndarray。
        支持类型：
        - numpy.ndarray
        - pandas DataFrame
        - pandas Series
        - list / tuple
        - 标量（int/float）
        """
        if isinstance(data, np.ndarray):
            return data
        if isinstance(data, (pd.DataFrame, pd.Series)):
            return data.to_numpy()
        if isinstance(data, (list, tuple)):
            return np.array(data)
        if np.isscalar(data):
            return np.array([data])
        raise TypeError(f"不支持的数据类型：{type(data)}")

    def to_torch(self, data):
        data_torch = torch.tensor(
            data,
            dtype=torch.float32,
            device=self.device
        )
        return data_torch

    def data_split(self, test_size=0.2, data_type='torch'):
        '''
        数据拆分，分为训练集和测试集
        :param test_size: 测试集比例
        :param data_type: 输出的数据类型：torch(default)/numpy
        :return:
        '''
        x_train, x_test, y_train, y_test = train_test_split(
            self.x,
            self.y,
            test_size=test_size,
            random_state=0
        )
        if data_type == 'torch':
            x_train = self.to_torch(x_train)
            x_test = self.to_torch(x_test)
            y_train = self.to_torch(y_train)
            y_test = self.to_torch(y_test)
        return x_train, x_test, y_train, y_test

    def kriging(self):
        '''
        Kriging代理模型
        :param params_output:   输出参数
        :return:
        '''
        model = kriging(
            self.x,
            self.y,
            testData=None,
            name='basic'
        )
        model.train()
        return model

    def kriging_cross_validation(self, test_size=0.2):
        '''
        Kriging代理模型交叉验证，给定输入输出参数，自动拆分训练集和测试集
        :param params_output:   输出参数
        :param test_size:       测试集比例
        :return:
        '''
        x_train, x_test, y_train, y_test = self.data_split(test_size=test_size, data_type='numpy')
        model = KRG(theta0=[1e-3])
        model.set_training_values(x_train, y_train)
        model.train()
        y_pred = model.predict_values(x_test)
        # 计算误差分析
        error_analysis = ErrorBasic(y_test, y_pred)
        error = error_analysis.evaluation_report()
        return model, error


if __name__ == '__main__':
    test = ModelTest()
    x, y = test.get_data(100, 2, -5, 5)
    model = SurrogateModelBasic(x, y)
    # kriging_model, error = model.kriging_cross_validation(0.2)