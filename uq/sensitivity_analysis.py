'''
@Desc:   敏感性分析
@Author: Dysin
@Time:   2024/9/26
'''
import numpy as np
from SALib.analyze import sobol
from SALib.sample import saltelli
from utils import Image

class Sensitivity_Analysis:
    def __init__(self, problem_params):
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

    def plt_image(self, path, file_name, si, text, text_size, text_position):
        '''
        绘制敏感度柱状图
        :param path:            文件路径
        :param file_name:       文件名
        :param si:              sobol.analyze
        :param text:            文本内容
        :param text_size:       文本字体大小
        :param text_position:   文本相对位置
        :return:
        '''
        x_list = []
        for i in range(len(self.names)):
            x_list.append(f'P{(i+1):02d}')
        image = Image(path, file_name, x_list, si['ST'], size=18)
        image.plt_bar(text, text_size, text_position)

if __name__ == '__main__':
    geometry_params_input_ranges = {
        '机身头部特征线首端点与x轴夹角': (40, 50),
        '机身头部特征线首端点斜率系数': (0.6, 1.2),
        '机身头部特征线末端点斜率系数': (0.6, 1.2),
        '机身后部特征线首端点与y轴夹角': (10, 30),
        '机身后部特征线末端点与x轴夹角': (5, 25),
        '机身后部特征线首端点斜率系数': (0.8, 1.1),
        '机身后部特征线末端点斜率系数': (0.8, 1.1),
        '机身侧向特征线1中点位置': (0.4, 0.55),
        '机身前部特征线相对长度Lower': (0.06, 0.12),
        '机身底部相对高度Lower': (0.03, 0.06),
        '机身底部后段相对长度Lower': (0.2, 0.25),
        '机身尾部特征线宽度系数Upper': (0.02, 0.04),
        '机身尾部特征线末端点与y轴夹角Upper': (5, 25),
        '机身横截面特征线1底部相对宽度Lower': (0.01, 0.03)
    }
    si = Sensitivity_Analysis(geometry_params_input_ranges)
    print(si.problem)