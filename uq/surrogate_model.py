'''
插值和预测模型，如：Kriging Model
Author: Dysin
Time:   2024.07.02
'''

import numpy as np
import pandas as pd
from pyKriging.krige import kriging
from smt.surrogate_models import KRG
from sklearn.model_selection import train_test_split
from utils import Image3D
from uq.error_analysis import BasicError

class SurrogateModel:
    def __init__(self, params_input, params_output):
        '''
        代理模型
        :param params_input:    输入参数
        :param params_output:   输出参数
        '''
        self.params_input = self.to_numpy(params_input)
        self.params_output = self.to_numpy(params_output)

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

    def kriging_model2(self):
        '''
        Kriging代理模型
        :param params_output:   输出参数
        :return:
        '''
        model = kriging(
            self.params_input,
            self.params_output,
            testData=None,
            name='basic'
        )
        model.train()
        return model

    def kriging_model(self):
        '''
        Kriging代理模型
        :param params_output:   输出参数
        :return:
        '''
        model = KRG(
            theta0=[1e-3]
        )
        model.set_training_values(self.params_input, self.params_output)
        model.train()
        return model

    def kriging_cross_validation(self, test_size):
        '''
        Kriging代理模型交叉验证，给定输入输出参数，自动拆分训练集和测试集
        :param params_output:   输出参数
        :param test_size:       测试集百分比
        :return:
        '''
        x_train, x_test, y_train, y_test = train_test_split(
            self.params_input,
            self.params_output,
            test_size=test_size,
            random_state=23
        )
        model = KRG(theta0=[1e-3])
        model.set_training_values(x_train, y_train)
        model.train()
        y_pred = model.predict_values(x_test)
        # 计算误差分析
        error_analysis = BasicError(y_test, y_pred)
        error = error_analysis.evaluation_report()
        return model, error

    def plt_kriging_surface(self, x_train, y_train, model, params_ranges):
        image3d = Image3D(12)
        train_points = np.column_stack((x_train[:, 0], x_train[:, 1], y_train))
        print(train_points)
        num_params = len(params_ranges)
        names = []
        ranges = []
        for key, value in params_ranges.items():
            names.append(key)
            ranges.append(value)
        x_list = []
        for i in range(num_params):
            x = np.linspace(ranges[i][0], ranges[i][1], 100)
            x_list.append(x)
        # 选择要绘制的 x 和 y 参数
        x_axis_index = 0  # 选择第 1 个输入参数（可以更改为 1-9）
        y_axis_index = 1  # 选择第 2 个输入参数（可以更改为 1-9）
        grid_x, grid_y = np.meshgrid(x_list[x_axis_index], x_list[y_axis_index])
        # 根据选择的 x 和 y 创建预测点
        # 将其他参数固定为其范围的中间值
        fixed_params = [(r[0] + r[1]) / 2 for r in ranges]  # 所有固定参数设置为范围中间值
        # 创建输入点网格
        grid_points = np.zeros((grid_x.size, num_params))
        grid_points[:, x_axis_index] = grid_x.ravel()  # 设置选定的 x 参数
        grid_points[:, y_axis_index] = grid_y.ravel()  # 设置选定的 y 参数
        # 为其他参数赋固定值
        for i in range(num_params):
            if i != x_axis_index and i != y_axis_index:
                grid_points[:, i] = fixed_params[i]
        grid_z = model.predict_values(grid_points)
        grid_z = grid_z.reshape(grid_x.shape)  # 将预测结果 reshape 为网格形状
        text = f'X: {names[0]}\nY: {names[1]}'
        image3d.scatters_and_surface(
            train_points,
            grid_x,
            grid_y,
            grid_z,
            text_position=[1, 1, 1],
            text=text
        )