'''
响应面法寻优
Author: Dysin
Time:   2024.06.01
'''

import numpy as np
import statsmodels.api as sm
from scipy.optimize import minimize

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class RSM:
    # 定义目标函数：最大化升阻比
    # 单目标优化
    def single_obj_opt(self, df_x, df_y, bound_values):
        df_x = sm.add_constant(df_x)
        model_y = sm.OLS(df_y, df_x).fit()
        def objective_function(params):
            params = [1] + list(params)  # 添加截距项
            ratio = model_y.predict([params])
            return -ratio  # 目标是最大化升阻比，所以取负值
        # 初始猜测
        initial_guess = np.mean(df_x.values[:, 1:], axis=0)
        result = minimize(objective_function, initial_guess, bounds=bound_values)

        optimal_params = result.x
        print("Optimal Parameters for Maximum Lift-to-Drag Ratio:", optimal_params)

    def multi_obj_opt(self, df_x, df_y, bound_values):
        df_x = sm.add_constant(df_x)
        model_ratio  = sm.OLS(df_y.iloc[:, 0], df_x).fit()
        model_vel    = sm.OLS(df_y.iloc[:, 1], df_x).fit()
        model_cl_avl = sm.OLS(df_y.iloc[:, 2], df_x).fit()
        def objective_function(params):
            params = [1] + list(params)  # 添加截距项
            max_ratio = model_ratio.predict(params)
            min_vel = model_vel.predict(params)
            max_cl_avl = model_cl_avl.predict(params)
            print(-max_ratio + min_vel - max_cl_avl)
            return -max_ratio + min_vel - max_cl_avl  # 目标是最大化升阻比，所以取负值
        # 初始猜测
        initial_guess = np.mean(df_x.values[:, 1:], axis=0)
        print(initial_guess)
        result = minimize(objective_function, initial_guess, bounds=bound_values)

        optimal_params = result.x
        print("Optimal Parameters for Maximum Lift-to-Drag Ratio:", optimal_params)

        # 绘制升阻比响应面图
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(df_x['巡航速度[m/s]'], df_x['外翼段梢根比'], df_y['升阻比'], c='r', marker='o')
        ax.set_xlabel('cl')
        ax.set_ylabel('lambda')
        ax.set_zlabel('ratio')
        plt.title('Response Surface of Lift/Drag Ratio')
        plt.show()