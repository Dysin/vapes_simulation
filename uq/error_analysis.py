"""
代理模型（Kriging/GPR）评估工具包（中文注释版）
包含：
- 基础误差指标 MAE / RMSE / MAPE / MaxError / R²
- 标准化残差（可检测异常点）
- LOOCV（逐点留一交叉验证）——计算 PRESS / Q²
- K-fold 交叉验证
- Gram/Cov 矩阵条件数检查（判断数值稳定性）
- 基于 sklearn 的 Kriging(GPR) 预测均值与不确定度
- 自带一套可运行的合成 CFD 示例数据
"""

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C
from numpy.linalg import cond
import warnings
warnings.filterwarnings("ignore")

class ErrorBasic:
    '''
    基础误差指标：
    MAE / RMSE / MAPE / MaxError / R²
    '''
    def __init__(self, y, y_pred):
        '''
        :param y: 实际值
        :param y_pred: 预测值
        '''
        self.y = y
        self.y_pred = y_pred

    def mae(self):
        '''
        平均绝对误差 (Mean Absolute Error)
        MAE: float
        '''
        return mean_absolute_error(self.y, self.y_pred)

    def rmse(self):
        '''
        均方根误差 (Root Mean Squared Error)
        :return:
        '''
        return np.sqrt(mean_squared_error(self.y, self.y_pred))

    def mape(self, eps=1e-8):
        '''
        平均绝对百分比误差 (Mean Absolute Percentage Error)
        :param eps: 防止除零的小常数
        :return:
        '''
        return np.mean(
            np.abs(
                (self.y - self.y_pred) /
                (np.where(np.abs(self.y) < eps, eps, self.y))
            )
        )

    def max_error(self):
        '''
        最大绝对误差
        :return:
        '''
        return np.max(np.abs(self.y - self.y_pred))

    def r2(self):
        '''
        决定系数 R²
        :return:
        '''
        return r2_score(self.y, self.y_pred)

    def standardized_residuals(self):
        """
        标准化残差 = (残差 - 均值) / 标准差
        用于发现异常点（|z| > 3）
        """
        e = self.y - self.y_pred
        se = np.std(e, ddof=1)
        if se == 0:
            return np.zeros_like(e)
        return (e - np.mean(e)) / se

    def evaluation_report(self):
        '''
        误差指标字典
        :param prefix:
        :return:
        '''
        res = {
            "MAE": self.mae(),
            "RMSE": self.rmse(),
            "MAPE": self.mape(),
            "R2": self.r2(),
            "MaxError": self.max_error()
        }
        return res

class CrossValidation(ErrorBasic):
    def __init__(self, y, y_pred):
        super().__init__(y, y_pred)

    def loocv_press_q2(self, X, y, model_builder):
        """
        留一交叉验证（LOOCV）
        对每个样本：
            - 删掉该点
            - 用剩余数据训练
            - 预测该点
        返回：
            PRESS = 所有 LOOCV 残差平方和
            Q2 = 1 - PRESS / SST
            preds = LOOCV 预测值
        """
        n = len(y)
        preds = np.zeros(n)
        for i in range(n):
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            X_train, y_train = X[mask], y[mask]
            X_test = X[~mask].reshape(1, -1)

            model = model_builder()
            model.fit(X_train, y_train)
            preds[i] = model.predict(X_test)[0]

        press = np.sum((y - preds) ** 2)
        q2 = 1 - press / np.sum((y - np.mean(y)) ** 2)
        return press, q2, preds

    def kfold_cv_metrics(self, X, y, model_builder, k=5):
        """
        K 折交叉验证
        返回平均指标
        """
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        preds = np.zeros_like(y, dtype=float)

        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train = y[train_idx]

            model = model_builder()
            model.fit(X_train, y_train)
            preds[test_idx] = model.predict(X_test)

        return {
            "MAE": self.mae(),
            "RMSE": self.rmse(),
            "MAPE": self.mape(),
            "R2": self.r2(),
            "MaxError": self.max_error()
        }

    def rbf_kernel_matrix(self, X1, X2, length_scale=1.0, sigma_f=1.0):
        """
        Gram 条件数
        简单 RBF kernel，用于构造 Gram 矩阵
        """
        X1 = np.atleast_2d(X1)
        X2 = np.atleast_2d(X2)
        dists = np.sum((X1[:, None, :] - X2[None, :, :]) ** 2, axis=2)
        K = sigma_f ** 2 * np.exp(-0.5 / length_scale ** 2 * dists)
        return K

    def condition_number_of_gram(self, X, kernel_func=None):
        """
        计算 Gram(或协方差) 矩阵的条件数
        条件数大 → 数值不稳定 → Kriging 可能奇异
        """
        if kernel_func is None:
            K = X.dot(X.T)
        else:
            K = kernel_func(X, X)

        K = 0.5 * (K + K.T)
        reg = 1e-12 * np.eye(K.shape[0])
        K_reg = K + reg

        try:
            kappa = cond(K_reg)
        except Exception:
            kappa = np.inf

        return kappa, K_reg

# --------------------------- 构建 Kriging (GPR) -----------------------------
def build_gpr_kernel():
    """
    最常用 Kriging 核函数：
    Constant * Matern(nu=2.5)
    """
    kernel = C(1.0, (1e-3, 1e3)) * Matern(
        length_scale=1.0,
        length_scale_bounds=(1e-3, 1e3),
        nu=2.5
    )
    return kernel

def model_builder_gpr(alpha=1e-6, n_restarts=0):
    """
    创建 GPR 模型实例
    alpha 影响噪声与数值稳定性
    """
    kernel = build_gpr_kernel()
    return GaussianProcessRegressor(
        kernel=kernel,
        alpha=alpha,
        normalize_y=True,
        random_state=0,
        n_restarts_optimizer=n_restarts
    )

def gpr_predict_with_uncertainty(X_train, y_train, X_pred, alpha=1e-6, n_restarts=0):
    """
    训练 Kriging 并返回：
    - 均值 y_mean
    - 标准差 y_std（不确定度）
    """
    gpr = model_builder_gpr(alpha=alpha, n_restarts=n_restarts)
    gpr.fit(X_train, y_train)
    y_mean, y_std = gpr.predict(X_pred, return_std=True)
    return gpr, y_mean, y_std

