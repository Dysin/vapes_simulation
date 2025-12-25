'''
插值和预测模型，如：Kriging Model
Author: Dysin
Time:   2024.07.02
'''
import sys

import torch
import gpytorch
import numpy as np
import pandas as pd
from pyKriging.krige import kriging
from smt.surrogate_models import KRG
from sklearn.model_selection import train_test_split
from utils.images_utils import PltImage3D
from uq.error_analysis import BasicError
from uq.model_test import ModelTest
import itertools

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
            random_state=23
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
        x_train, x_test, y_train, y_test = self.data_split(
            test_size=test_size,
            data_type='numpy'
        )
        model = KRG(theta0=[1e-3])
        model.set_training_values(x_train, y_train)
        model.train()
        y_pred = model.predict_values(x_test)
        # 计算误差分析
        error_analysis = BasicError(y_test, y_pred)
        error = error_analysis.evaluation_report()
        return model, error

    def gaussian_process(self, test_size=0.2, training_iter=300):
        """
        高斯过程，多输出代理模型交叉验证
        每一个输出量 -> 一个单输出 GP（Exact GP, GPU）

        Returns
        -------
        models : list
            每个输出对应一个 GPModel
        errors : list
            每个输出对应一个误差评估结果
        """
        # ---------- 数据拆分 ----------
        x_train, x_test, y_train, y_test = self.data_split(
            test_size=test_size
        )
        # ---------- 关键：统一输出维度 ----------
        # 若为单输出：(N,) -> (N, 1)
        if y_train.ndim == 1:
            y_train = y_train.unsqueeze(1)
            y_test = y_test.unsqueeze(1)
        n_outputs = y_train.shape[1]
        models = []
        likelihoods = []
        errors = []
        # ---------- 每个输出单独训练一个 GP ----------
        for i in range(n_outputs):
            # 取第 i 个输出（关键）
            y_train_i = y_train[:, i]
            y_test_i = y_test[:, i]
            # ---------- GP 定义 ----------
            likelihood = gpytorch.likelihoods.GaussianLikelihood(
            ).to(self.device)
            model = GPModel(
                train_x=x_train,
                train_y=y_train_i,
                likelihood=likelihood
            ).to(self.device)
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
            likelihoods.append(likelihood)
            errors.append(error)
        return models, likelihoods, errors

    def plot_gp_surface(
            self,
            model,
            likelihood,
            path=None,
            output_idx=0,
            n_grid=50
    ):
        '''
        绘制 GPyTorch 模型的二维输入 - 单输出响应曲面，并可绘制训练点
        :param model:       已训练好的 GP 模型
        :param likelihood:  对应的似然
        :param path:        如果指定路径，则保存图像
        :param output_idx:  要绘制的输出索引
        :param n_grid:      网格密度
        :return:
        '''
        # 获取两两组合，不重复
        combinations = list(itertools.combinations(range(self.x.shape[1]), 2))
        print(combinations)
        for i_comb in range(len(combinations)):
            x_idx, y_idx = combinations[i_comb]
            # 构建网格
            x_lin = np.linspace(
                self.x_tensor[:, x_idx].min().item(),
                self.x_tensor[:, x_idx].max().item(),
                n_grid
            )
            y_lin = np.linspace(
                self.x_tensor[:, y_idx].min().item(),
                self.x_tensor[:, y_idx].max().item(),
                n_grid
            )
            X1, X2 = np.meshgrid(x_lin, y_lin)

            # 生成网格输入
            X_grid = np.zeros(
                (n_grid * n_grid, self.x_tensor.shape[1]),
                dtype=np.float32
            )
            X_grid[:, x_idx] = X1.ravel()
            X_grid[:, y_idx] = X2.ravel()
            for i in range(self.x_tensor.shape[1]):
                if i != x_idx and i != y_idx:
                    X_grid[:, i] = self.x_tensor[:, i].mean().item()
            X_grid = self.to_torch(X_grid)

            # 确保模型在同一设备
            model = model.to(self.device)
            likelihood = likelihood.to(self.device)

            # 模型预测
            model.eval()
            likelihood.eval()
            with torch.no_grad(), torch.inference_mode():
                pred = likelihood(model(X_grid))
                y_mean = pred.mean
                # 自动判断单输出/多输出
                if y_mean.ndim == 1:
                    Y_pred = y_mean.reshape(n_grid, n_grid).cpu().numpy()
                else:
                    Y_pred = y_mean[:, output_idx].reshape(n_grid, n_grid).cpu().numpy()

            # 绘制曲面
            plt3d = PltImage3D(
                path = path,
                image_name = f'surrogate_model_gp{output_idx+1:02d}_{i_comb+1:02d}'
            )
            axis_labels = [
                f'X{x_idx+1}',
                f'X{y_idx+1}',
                f'Y{output_idx+1}'
            ]
            plt3d.scatters_and_surface(
                self.x[:, x_idx],
                self.x[:, y_idx],
                self.y[:, output_idx],
                X1,
                X2,
                Y_pred,
                axis_labels=axis_labels,
            )


if __name__ == '__main__':
    test = ModelTest()
    x, y = test.get_data(100, 2, -5, 5)
    model = SurrogateModel(x, y)
    # kriging_model, error = model.kriging_cross_validation(0.2)
    gp_model, error = model.gaussian_process(0.2)
    print(f'[INFO] Number of model: {len(gp_model)}')
    print(error)