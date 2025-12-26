'''
@Desc:   基于高斯过程回归代理模型 + 分析
@Author: Dysin
@Date:   2025/12/25
'''

import torch
import gpytorch
import numpy as np
import itertools
from sklearn.model_selection import KFold
from SALib.sample import saltelli
from SALib.analyze import sobol
from uq.surrogate_model import SurrogateModelBasic
from uq.error_analysis import ErrorBasic
from uq.sensitivity_analysis import SensitivityAnalyzer
from utils.images_utils import PlotImage3D

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

class GPAnalyzer(SurrogateModelBasic):
    """
    高斯过程回归代理模型 + 分析器
    功能：
    1. GP 训练
    2. 预测 & 不确定性估计
    3. 误差分析（训练/测试集）
    4. K折交叉验证
    5. Sobol 敏感性分析
    """

    def __init__(self, params_input, params_output):
        """
        初始化 GPAnalyzer
        参数：
        - X : ndarray (N, d)  输入参数
        - y : ndarray (N,)    输出参数
        - var_names : list    参数名称，用于敏感性分析显示，默认 x0,x1,...
        - test_size : float   测试集比例，默认 0.2
        - device : str        'cpu' 或 'cuda'，支持 GPU
        """
        super().__init__(params_input, params_output)

    # ------------------ 模型训练 ------------------
    def train_model(self, X, y, tol=1e-4, max_iter=10000, lr=0.1):
        """
        训练 GP 模型
        参数：
        - training_iter : int   迭代次数
        - lr : float            学习率
        """
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood(
        ).to(self.device)
        self.model = GPModel(
            train_x=X,
            train_y=y,
            likelihood=self.likelihood
        ).to(self.device)
        # Adam 优化器
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        # 对数边际似然函数（最大化）
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
        prev_loss = None
        for i in range(max_iter):
            optimizer.zero_grad()
            output = self.model(X)
            loss = -mll(output, y)
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                print(f"[GP Train] Iter {i:4d} | Loss = {loss.item():.4f}")
            cur_loss = loss.item()
            if prev_loss is not None:
                rel_change = abs(prev_loss - cur_loss) / (abs(prev_loss) + 1e-12)
                if rel_change < tol and i > 200:
                    print(f"[GP Train] Early stop at iter {i}, "
                          f"rel_change={rel_change:.2e}")
                    break
            prev_loss = cur_loss
        return self.model, self.likelihood

    # ------------------ 预测 ------------------
    def predict(self, X):
        """
        对新样本进行预测
        输入：
        - X : ndarray or torch.Tensor (M, d) 待预测样本
        输出：
        - mean : 预测均值 (M,)
        - std  : 预测标准差 (M,) → 不确定性
        """
        self.model.eval()
        self.likelihood.eval()
        X = self.to_torch(X)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred = self.likelihood(self.model(X))
        mean = pred.mean.cpu().numpy()
        std = pred.variance.sqrt().cpu().numpy()
        return mean

    # ------------------ K折交叉验证 ------------------
    def cross_validate(self, X, y, n_splits=5, training_iter=200):
        """
        K折交叉验证
        输出：
        - results : list[dict] 每折误差
        """
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)
        results = []

        for i, (tr, te) in enumerate(kf.split(X)):
            # 划分训练/测试
            X_tr = self.to_torch(X[tr])
            y_tr = self.to_torch(y[tr])
            X_te = self.to_torch(X[te])
            y_te = self.to_torch(y[te])

            # 创建临时 GP 模型
            likelihood = gpytorch.likelihoods.GaussianLikelihood(
            ).to(self.device)
            model = GPModel(
                train_x=X,
                train_y=y,
                likelihood=self.likelihood
            ).to(self.device)
            model.device = self.device
            model.X_train = X_tr
            model.y_train = y_tr
            model.mean_module = gpytorch.means.ConstantMean()
            model.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
            model.likelihood = likelihood
            model.train()
            likelihood.train()

            optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
            for _ in range(training_iter):
                optimizer.zero_grad()
                loss = -mll(model(X_tr), y_tr)
                loss.backward()
                optimizer.step()
            # 预测
            model.eval()
            likelihood.eval()
            with torch.no_grad():
                y_pred = likelihood(model(X_te)).mean
            metrics = ErrorBasic(y_te, y_pred)
            results.append(metrics)
            print(f"[CV Fold {i}] {metrics}")
        return results

    def plot_surface(
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
            plt3d = PlotImage3D(
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

    def workflow(self, path_image, problem_params=None, test_size=0.2):
        x_train, x_test, y_train, y_test = self.data_split(test_size)
        # 统一输出维度
        if y_train.ndim == 1:
            y_train = y_train.unsqueeze(1)
            y_test = y_test.unsqueeze(1)
        n_outputs = y_train.shape[1]
        models = []
        likelihoods = []
        errors = []
        for i in range(n_outputs):
            # 取第 i 个输出（关键）
            y_train_i = y_train[:, i]
            y_test_i = y_test[:, i].cpu().numpy()
            model, likelihood = self.train_model(x_train, y_train_i)
            y_pred_i = self.predict(x_test)
            error_analyzer = ErrorBasic(y_test_i, y_pred_i)
            error = error_analyzer.evaluation_report()
            models.append(model)
            likelihoods.append(likelihood)
            errors.append(error)

            # 绘制代理模型曲面
            self.plot_surface(
                model,
                likelihood,
                path=path_image,
                output_idx=i
            )

            if problem_params is not None:
                # 敏感性分析
                sensitivity_analyzer = SensitivityAnalyzer(problem_params)
                si_x = sensitivity_analyzer.get_params_input(100)
                si_y = self.predict(si_x)
                si = sensitivity_analyzer.solve(si_y)
                sensitivity_analyzer.plt_image(
                    path=path_image,
                    file_name=f'sensitivity_{i+1:02d}',
                    si=si
                )