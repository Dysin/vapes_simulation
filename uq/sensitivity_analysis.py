'''
@Desc:   敏感性分析
@Author: Dysin
@Time:   2024/9/26
'''
import numpy as np
from SALib.analyze import sobol
from SALib.sample import saltelli
from utils.images_utils import PlotImage2D

class SensitivityAnalyzer:
    '''
    敏感性分析器
    适用于：小样本 Kriging / GPR 代理模型
    '''
    def __init__(self, problem_params):
        '''
        trained surrogate model
        :param model:
        :param problem_params:
        '''
        self.names = []
        ranges = []
        for key, value in problem_params.items():
            self.names.append(key)
            ranges.append(value)
        self.problem = {
            'num_vars': len(problem_params),
            'names': self.names,
            'bounds': ranges
        }

    def get_params_input(self, sample_num):
        params_input = saltelli.sample(self.problem, sample_num)
        return params_input

    def solve(self, params_output):
        si = sobol.analyze(
            self.problem,
            params_output,
            print_to_console=True
        )
        # 输出敏感性分析结果
        print("First-order sensitivity indices:")
        print(si['S1'])
        print("Total-order sensitivity indices:")
        print(si['ST'])
        return si

    def plt_image(self, path, file_name, si):
        '''
        绘制敏感度柱状图
        :param path:            文件路径
        :param file_name:       文件名
        :param si:              sobol.analyze
        :return:
        '''
        y_dict = {
            'S1': si['S1'],
            'ST': si['ST'],
        }
        plt_image = PlotImage2D(
            path,
            file_name,
            font_size=10
        )
        plt_image.grouped_bar(
            categories=self.names,
            data_dict=y_dict,
            show_values=False
        )