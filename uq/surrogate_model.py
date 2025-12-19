'''
插值和预测模型，如：Kriging Model
Author: Dysin
Time:   2024.07.02
'''

import torch
import gpytorch
import numpy as np
import pandas as pd
from pyKriging.krige import kriging
from smt.surrogate_models import KRG
from sklearn.model_selection import train_test_split
from utils import Image3D
from uq.error_analysis import BasicError
from uq.model_test import ModelTest

class GPModel(gpytorch.models.ExactGP):
    '''
    基于 GPyTorch 的高斯过程回归模型（Exact GP）
    等价于 Kriging 代理模型
    '''
    def __init__(self, train_x, train_y, likelihood):
        '''

        :param train_x: torch.Tensor, shape = (N, d)
                        训练输入样本
                        例如：
                        N = CFD 仿真样本数
                        d = 气道参数维度（入口直径、咪头直径、吸嘴直径等）
        :param train_y: torch.Tensor, shape = (N,)
                        训练输出
                        例如：
                        压损、最大速度、湍流强度、口感指标等
        :param likelihood:  gpytorch.likelihoods.GaussianLikelihood
                            高斯似然函数
                            用于描述 CFD 结果中的数值噪声 / 仿真误差
        '''
        # ---------- 父类初始化（必须） ----------
        # ExactGP 内部会保存：
        # - 训练数据 train_x / train_y
        # - likelihood
        # - 用于计算对数边际似然（mll）
        super().__init__(train_x, train_y, likelihood)

        # ---------- 均值函数（Mean Function） ----------
        # ConstantMean 表示：
        #   y(x) = 常数 + 波动
        #
        # 工程理解：
        # CFD 结果往往围绕一个基准水平上下变化，
        # 不假设显式线性/非线性趋势，让核函数建模变化
        self.mean_module = gpytorch.means.ConstantMean()

        # ---------- 协方差函数（Kernel / Covariance） ----------
        # 核函数是 GP/Kriging 的“灵魂”
        #
        # RBFKernel（高斯核）假设：
        # - 输入参数变化是连续的
        # - 相近几何参数 → 相近流动结果
        #
        # ScaleKernel 的作用：
        # - 在 RBFKernel 外层乘一个 σ²（输出尺度）
        # - 等价于 Kriging 中的 process variance
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel()
        )

    def forward(self, x):
        '''
        GP 的前向传播
        给定输入 x，输出一个多元正态分布
        :param x:   torch.Tensor, shape = (M, d)
                    待预测样本（新气道几何参数）
        :return: MultivariateNormal
                - mean : 预测均值
                - covariance : 协方差矩阵（不确定度来源）
        '''

        # ---------- 计算均值 ----------
        # 对每一个输入样本，给出预测均值
        mean_x = self.mean_module(x)

        # ---------- 计算协方差 ----------
        # 反映不同样本之间的相关性
        covar_x = self.covar_module(x)

        # ---------- 返回多元高斯分布 ----------
        return gpytorch.distributions.MultivariateNormal(
            mean_x,
            covar_x
        )

class SurrogateModel:
    def __init__(self, params_input, params_output):
        '''
        代理模型
        :param params_input:    输入参数
        :param params_output:   输出参数
        '''
        self.params_input = self.to_numpy(params_input)
        self.params_output = self.to_numpy(params_output)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    def kriging(self):
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

    def kriging_cross_validation(self, test_size=0.2):
        '''
        Kriging代理模型交叉验证，给定输入输出参数，自动拆分训练集和测试集
        :param params_output:   输出参数
        :param test_size:       测试集比例
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

    def gp_cross_validation(self, test_size=0.2, training_iter=300):
        """
        多输出代理模型交叉验证
        每一个输出量 -> 一个单输出 GP（Exact GP, GPU）

        Returns
        -------
        models : list
            每个输出对应一个 GPModel
        errors : list
            每个输出对应一个误差评估结果
        """
        # ---------- 数据拆分 ----------
        x_train, x_test, y_train, y_test = train_test_split(
            self.params_input,
            self.params_output,
            test_size=test_size,
            random_state=23
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # ---------- numpy -> torch ----------
        x_train = torch.tensor(x_train, dtype=torch.float32, device=device)
        x_test = torch.tensor(x_test, dtype=torch.float32, device=device)
        y_train = torch.tensor(y_train, dtype=torch.float32, device=device)
        y_test = torch.tensor(y_test, dtype=torch.float32, device=device)
        # ---------- 关键：统一输出维度 ----------
        # 若为单输出：(N,) -> (N, 1)
        if y_train.ndim == 1:
            y_train = y_train.unsqueeze(1)
            y_test = y_test.unsqueeze(1)
        n_outputs = y_train.shape[1]
        models = []
        errors = []
        # ---------- 每个输出单独训练一个 GP ----------
        for i in range(n_outputs):
            # 取第 i 个输出（关键）
            y_train_i = y_train[:, i]
            y_test_i = y_test[:, i]
            # ---------- GP 定义 ----------
            likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
            model = GPModel(
                train_x=x_train,
                train_y=y_train_i,
                likelihood=likelihood
            ).to(device)
            # ---------- 训练 ----------
            model.train()
            likelihood.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
            for _ in range(training_iter):
                optimizer.zero_grad()
                output = model(x_train)
                loss = -mll(output, y_train_i)
                loss.backward()
                optimizer.step()
            # ---------- 预测 ----------
            model.eval()
            likelihood.eval()
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                y_pred_i = likelihood(model(x_test)).mean
            # ---------- 误差分析 ----------
            y_pred_i = y_pred_i.cpu().numpy()
            y_test_i = y_test_i.cpu().numpy()
            error_analysis = BasicError(y_test_i, y_pred_i)
            error = error_analysis.evaluation_report()
            models.append(model)
            errors.append(error)
        return models, errors

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


if __name__ == '__main__':
    test = ModelTest()
    x, y = test.get_data(100, 2, -5, 5)
    model = SurrogateModel(x, y)
    # kriging_model, error = model.kriging_cross_validation(0.2)
    gp_model, error = model.gp_cross_validation(0.2)
    print(f'[INFO] Number of model: {len(gp_model)}')
    print(error)