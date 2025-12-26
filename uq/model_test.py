'''
@Desc:   用来测试代理模型的函数、数据等
@Author: Dysin
@Date:   2025/12/18
'''

import numpy as np
from utils.images_utils import PlotImage3D

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class ModelTest:
    def __init__(self):
        print('[INFO] UQ Model Test')

    def ackley(self, x, a=20, b=0.2, c=2 * np.pi):
        '''
        Ackley function
        :param x: ndarray, shape (n,) or (N, n)，输入变量
        :param a: Ackley 函数参数
        :param b: Ackley 函数参数
        :param c: Ackley 函数参数
        :return: Ackley 函数值
        '''
        x = np.atleast_2d(x)
        n = x.shape[1]
        sum_sq = np.sum(x ** 2, axis=1)
        sum_cos = np.sum(np.cos(c * x), axis=1)
        term1 = -a * np.exp(-b * np.sqrt(sum_sq / n))
        term2 = -np.exp(sum_cos / n)
        y = term1 + term2 + a + np.e
        return y

    def get_data(self, n_samples, dim, lower, upper):
        # 随机种子
        np.random.seed(42)
        x_tensor = np.random.uniform(lower, upper, size=(n_samples, dim))
        # 输出 y
        y = self.ackley(x_tensor)
        return x_tensor, y

    def ackley2(self, x, y, a=20, b=0.2, c=2 * np.pi):
        """
        2D Ackley function
        """
        term1 = -a * np.exp(-b * np.sqrt(0.5 * (x ** 2 + y ** 2)))
        term2 = -np.exp(0.5 * (np.cos(c * x) + np.cos(c * y)))
        return term1 + term2 + a + np.e

    def plt_image(self):
        # 定义范围
        x = np.linspace(-10, 10, 400)
        y = np.linspace(-10, 10, 400)

        X, Y = np.meshgrid(x, y)
        print(len(X))
        Z = self.ackley2(x, y)

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")

        ax.plot_surface(X, Y, Z, rstride=5, cstride=5)

        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_zlabel("f(x)")
        ax.set_title("Ackley Function (3D Surface)")

        plt.tight_layout()
        plt.show()

        # plt3d = PlotImage3D(22)
        # points = plt3d.merge_to_3cols(x, y, Z)
        # print(points)
        # plt3d.scatters_and_surface(
        #     points,
        #     None,
        #     None,
        #     None
        # )


if __name__ == '__main__':
    test = ModelTest()
    x, y = test.get_data(1000, 2, -5, 5)
    # plt_image = PlotImage3D(22)
    # points = plt_image.merge_to_3cols(x, y)
    # plt_image.scatters_and_surface(
    #     points,
    #     None,
    #     None,
    #     None
    # )
    # print(points)
    test.plt_image()