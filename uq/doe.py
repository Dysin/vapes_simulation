'''
设计实验（Design of Experiments，DOE）数据分析方法
Author: Dysin
Time:   2024.05.29
'''

import numpy as np
from pyDOE import lhs

class DOE:
    def __init__(self, params_range, sample_num):
        self.params_range = np.array(params_range)  # Convert to NumPy array for easier manipulation
        self.sample_num = sample_num

    # 拉丁超立方采样
    def latin_hypercube_sampling(self):
        '''
        高维空间采样
        '''
        # 拉丁超立方采样
        lhs_design = lhs(len(self.params_range), samples=self.sample_num)

        # 通过广播将参数范围映射到样本空间
        scaled_samples = (lhs_design * (
            self.params_range[:, 1] -
            self.params_range[:, 0]) +
            self.params_range[:, 0]
        )

        return scaled_samples  # Transpose to return each row as a parameter set

if __name__ == '__main__':
    input_params_range = [
        [-0.2, 0.4],
        [-0.2, 0],
        [-0.2, 0.2]
    ]
    doe = DOE(input_params_range, 100)
    res = doe.latin_hypercube_sampling()
    print(res)
    print(res[0][0], res[0][1], res[0][2])