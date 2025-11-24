'''
@Desc:   绘制STARCCM的检测曲线等
@Author: Dysin
@Date:   2025/10/15
'''

from utils.data_manager import CSV
from utils.files import Image

class StarCCMDataAnalysis():
    def __init__(self, path):
        self.path = path

    def plt_curve(
        self,
        file_name
    ):
        csv = CSV(self.path, file_name)
        df = csv.read()
        image_manager = Image(
            self.path,
            file_name,
            x=df.iloc[:,0],
            y=df.iloc[:,1],
            label_x='Iterations',
            label_y='Pressure'
        )
        image_manager.plt_curve()

    def get_pressure_difference_by_p(self, file_name1, file_name2):
        csv1 = CSV(self.path, file_name1)
        df1 = csv1.read()
        csv2 = CSV(self.path, file_name2)
        df2 = csv2.read()
        delta = df1.iloc[-1,1] - df2.iloc[-1,1]
        print(f'[INFO] {file_name1} 和 {file_name2} 的压力差为{delta}')
        return delta

    def get_value(self, file_name):
        '''
        获取CSV表最后一行的值
        :param path:
        :param file_name:
        :return:
        '''
        csv = CSV(self.path, file_name)
        df = csv.read()
        print(f'[INFO] {file_name} 数值为{df.iloc[-1,1]}')
        return df.iloc[-1,1]

if __name__ == '__main__':
    path = r'D:\1_Work\active\202510_ATN021\simulation\grid_independence\level_2'
    starccm_post = StarCCMDataAnalysis(path)
    # plt_starccm_curve(path, 'p_in')
    # plt_starccm_curve(path, 'p_out')
    # plt_starccm_curve(path, 'p_sensor')
    starccm_post.get_value('p_in')