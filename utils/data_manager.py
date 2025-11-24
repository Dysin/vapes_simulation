'''
数据处理
Author: Dysin
Time:   2024.05.24
'''
import math
import os
import re
import numpy as np
import pandas as pd
import csv

class Common_Interface:
    '''
    数据通用接口
    :param path:    文件路径
    '''
    def __init__(self, path=None):
        self.path = path

    def normalize(self, data, inverse=False):
        '''
        标准化函数
        :param data:    某列特征数据
        :param inverse: 是否反转，一般用于值越小，分数越高的情况
        :return:
        '''
        epsilon = 1e-12
        data = np.array(data)
        if inverse:
            data_output = (data.max() - data) / (data.max() - data.min() + epsilon)
        else:
            data_output = (data - data.min()) / (data.max() - data.min() + epsilon)
        return data_output

class CSV:
    def __init__(self, path, file_name):
        '''
        csv数据
        :param path:        文件路径
        :param file_name:   文件名
        '''
        self.path = path
        self.file_name = file_name
        self.file = os.path.join(self.path, f'{self.file_name}.csv')

    def read(self):
        '''
        以pandas形式获取csv数据
        :return:
        '''
        df = pd.read_csv(self.file)
        return df

    def read_data(self):
        '''
        以列表形式获取csv每行数据
        :return:
        '''
        data = []
        # 打开并读取 CSVUtils 文件
        with open(self.file, mode='r') as file:
            reader = csv.reader(file)
            # 遍历并打印每一行
            for row in reader:
                data.append(row)

    def write_row_data(self, row_data):
        '''
        将一行数据追加到csv中
        :param row_data:
        :return:
        '''
        file_csv = os.path.join(self.path, f'{self.file_name}.csv')
        # 打开或创建一个CSV文件，并将一行数据写入其中
        with open(file_csv, mode='a', newline='') as file:
            writer = csv.writer(file)
            # 写入一行数据
            writer.writerow(row_data)

    def write_data(self, data):
        '''
        写入整个pandas数据
        :param data: pandas数据
        :return:
        '''
        file_csv = os.path.join(self.path, f'{self.file_name}.csv')
        data.to_csv(file_csv, index=False)

    def remove(self):
        '''
        删除文件
        :return:
        '''
        try:
            os.remove(self.file)
        except:
            print(f'File {self.file} deleted')

    def data_drop_duplicates(self, data):
        '''
        去掉重复行
        :param data:  pandas数据
        :return:
        '''
        df_head = data.iloc[:1]
        df_drop = data.drop_duplicates()
        df_drop = df_drop.drop_duplicates().reset_index(drop=True)
        return df_drop

    def sort(self, data):
        '''
        按第一列排序
        :param data:    数据
        :return:
        '''
        df = data.sort_values(by='A')
        return df