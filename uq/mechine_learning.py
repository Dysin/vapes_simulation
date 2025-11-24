'''
机器学习方法寻优
Author: Dysin
Time:   2024.06.01
'''

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.optimize import minimize

class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNN, self).__init__()
        self.layer1 = nn.Linear(input_dim, 64)
        self.layer2 = nn.Linear(64, 64)
        self.output = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.output(x)
        return x

class ML:
    def train_model(self, model, x_train, y_train, num_epochs=100, learning_rate=0.001):
        criterion = nn.MSELoss()                                        # 损失函数：均方误差
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)    # 优化器：Adam
        for epoch in range(num_epochs):
            model.train()                                               # 设置模型为训练模式
            optimizer.zero_grad()                                       # 清空梯度
            outputs = model(x_train)                                    # 前向传播
            loss = criterion(outputs, y_train)                          # 计算损失
            loss.backward()                                             # 反向传播
            optimizer.step()                                            # 更新权重
    def neural_networks(self, df_x, df_y, params_bound):
        # 1.将数据分成训练集和测试集
        x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2, random_state=0)

        # 2.对数据进行标准化处理
        standard_x = StandardScaler()
        standard_y = StandardScaler()

        x_train_standard = standard_x.fit_transform(x_train)
        x_test_standard = standard_x.transform(x_test)
        y_train_standard = standard_y.fit_transform(y_train.values.reshape(-1, 1))

        # 将数据转换为PyTorch张量
        x_train_tensor = torch.tensor(x_train_standard, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train_standard, dtype=torch.float32)

        # 3.构建神经网络模型
        input_dim = x_train.shape[1]
        print(input_dim)
        model_y = SimpleNN(input_dim)

        # 4.训练模型
        self.train_model(model_y, x_train_tensor, y_train_tensor)

        # 5.定义优化问题
        def object_func(params):
            params_df = pd.DataFrame([params], columns=df_x.keys())
            params_standard = standard_x.transform(params_df)
            params_tensor = torch.tensor(params_standard, dtype=torch.float32)

            model_y.eval()  # 设置模型为评估模式

            with torch.no_grad():
                y_pred = model_y(params_tensor).item()

            y_pred_unstand = standard_y.inverse_transform([[y_pred]])[0][0]

            return -y_pred_unstand

        # 6.执行优化
        initial_guess = np.mean(df_x.values[:, 0:], axis=0) # 初始猜测值
        # initial_guess = [32, 0.55, 0.7, 0.6, 0.6]
        print(initial_guess)
        result = minimize(object_func, initial_guess, bounds=params_bound, method='SLSQP')

        optimal_params = result.x
        optimal_params_df = pd.DataFrame([optimal_params], columns=df_x.keys())
        optimal_params_standard = standard_x.transform(optimal_params_df)
        optimal_params_tensor = torch.tensor(optimal_params_standard, dtype=torch.float32)

        model_y.eval()

        with torch.no_grad():
            optimal_y_standard = model_y(optimal_params_tensor).item()

        optimal_y = standard_y.inverse_transform([[optimal_y_standard]])[0][0]

        print(f'最佳输入参数：')
        for i, param_name in enumerate(df_x.keys()):
            print(f'{param_name}: {optimal_params[i]}')
        print(f'最佳升阻比: {optimal_y}')
        # print(f'最佳平飞速度: {optimal_speed}')