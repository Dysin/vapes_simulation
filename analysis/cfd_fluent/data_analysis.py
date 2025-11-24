'''
@Desc:   Fluent结果数据分析
@Author: Dysin
@Time:   2024/9/22
'''

import os
import numpy as np
from utils import Image
from utils import Excel
from utils import Files
from utils import Common_Interface

class Fluent_Data_Analysis:
    def __init__(self, path):
        '''
        Fluent结果数据分析，如升阻力曲线图等
        :param path: 文件路径
        '''
        self.path = path

    def read_fluent_dat(self, file_name):
        '''
        读取Fluent data
        :param file_name:   dat文件名
        :return:
        '''
        file = open(os.path.join(self.path, f'{file_name}.dat'), 'r')
        lines = file.readlines()
        file.close()
        return lines

    def get_last_iters_data(self, file_name) -> float:
        '''
        获取最后一个迭代步的力或力矩
        :param file_name: 文件名
        :return:
        '''
        data = self.read_fluent_dat(file_name)[-1]
        value = float(data.split()[-1])
        return value

    def write_excel(self, angles, angle_name):
        '''
        将不同攻角或不同侧滑角的力和力矩写入excel
        :param angles: 角度列表
        :param angle_name:  'aoa'/'beta'
        :return:
        '''
        excel = Excel(self.path, f'forces_vs_{angle_name}')
        workbook, sheet = excel.new_sheet()
        header = [
            'AOA',
            'Lift [N]',
            'Drag [N]',
            'Lift Coefficient',
            'Drag Coefficient',
            'Moment Coefficient',
            'Lift-to-Drag Ratio'
        ]
        excel.creat_title(sheet, header)
        for i in range(len(angles)):
            lift = self.get_last_iters_data(f'lift_aoa{angles[i]}_beta0')
            drag = self.get_last_iters_data(f'drag_aoa{angles[i]}_beta0')
            cl = self.get_last_iters_data(
                f'lift_coefficient_aoa{angles[i]}_beta0'
            )
            cd = self.get_last_iters_data(
                f'drag_coefficient_aoa{angles[i]}_beta0'
            )
            cm = self.get_last_iters_data(
                f'moment_coefficient_aoa{angles[i]}_beta0'
            )
            row_data = [
                angles[i],
                lift,
                drag,
                cl,
                cd,
                cm,
                lift / (drag + 1e-9)
            ]
            excel.write_vlaues(sheet, row_data, i)
        excel.style(sheet, 12)
        excel.save(workbook)

    def write_excel_comprehensive(self, path_save):
        '''
        输出综合分析表格
        :param path_save:   表格保存路径
        :return:
        '''
        excel_comprehensive = Excel(path_save, f'forces_comprehensive')
        workbook, sheet = excel_comprehensive.new_sheet()
        header = [
            # 'Geometry name',
            'Lift coefficient at zero AOA',
            'Lift-to-Drag ratio at zero AOA',
            'AOA at maximum lift coefficient',
            'Score'
        ]
        excel_comprehensive.creat_title(sheet, header)
        comi = Common_Interface()
        # 权重
        weights = [0.3, 0.5, 0.2]
        cl0_list = []
        ld0_list = []
        max_aoa_list = []
        for i in range(len(self.path)):
            excel = Excel(self.path[i], 'forces_vs_aoa')
            data = excel.read(type='float')
            cl_at_aoa0 = None
            ld_ratio_at_aoa0 = None
            aoa_at_max_cl = None
            for j in range(len(data)):
                if data.loc[j, 'AOA'] == 0:
                    cl_at_aoa0 = data.loc[j, 'Lift Coefficient']
                    ld_ratio_at_aoa0 = data.loc[j, 'Lift-to-Drag Ratio']
                if data.loc[j, 'Lift Coefficient'] == max(data.loc[:, 'Lift Coefficient']):
                    aoa_at_max_cl = data.loc[j, 'AOA']
            cl0_list.append(cl_at_aoa0)
            ld0_list.append(ld_ratio_at_aoa0)
            max_aoa_list.append(aoa_at_max_cl)
        ndh_cl0_list = comi.normalize(cl0_list)
        ndh_ld0_list = comi.normalize(ld0_list)
        ndh_max_aoa_list = comi.normalize(max_aoa_list)
        for i in range(len(cl0_list)):
            score = weights[0] * ndh_cl0_list[i] + weights[1] * ndh_ld0_list[i] + weights[2] * ndh_max_aoa_list[i]
            row_data = [
                cl0_list[i],
                ld0_list[i],
                max_aoa_list[i],
                score
            ]
            excel_comprehensive.write_vlaues(sheet, row_data, i)
        excel_comprehensive.style(sheet, 12)
        excel_comprehensive.save(workbook)

    def image_forces_vs_aoa(self):
        excel = Excel(self.path, 'forces_vs_aoa')
        data = excel.read(type='float')
        header = data.columns.tolist()
        aoas = data.loc[:, header[0]]
        file_names = [
            'lift',
            'drag',
            'lift_coefficient',
            'drag_coefficient',
            'moment_coefficient',
            'lift_to_drag_ratio'
        ]
        for i in range(1, len(header)):
            forces = data.loc[:, header[i]]
            image = Image(
                self.path,
                f'{file_names[i-1]}_vs_aoa',
                aoas,
                forces,
                label_x=header[0],
                label_y=header[i],
                size=18
            )
            image.plt_curve()

    def image_forces_vs_aoa_comprehensive(self, path_save, legends, legend_size):
        aoas_list = []
        file_names = [
            'lift',
            'drag',
            'lift_coefficient',
            'drag_coefficient',
            'moment_coefficient',
            'lift_to_drag_ratio'
        ]
        forces_list = [[] for _ in range(len(file_names))]
        labels_x = []
        labels_y = []
        for i in range(len(self.path)):
            excel = Excel(self.path[i], 'forces_vs_aoa')
            data = excel.read(type='float')
            header = data.columns.tolist()
            aoas = data.loc[:, header[0]]
            aoas_list.append(aoas)
            labels_x.append(header[0])
            for j in range(1, len(header)):
                forces = data.loc[:, header[j]]
                forces_list[j-1].append(forces)
                labels_y.append(header[j])
        for i in range(len(file_names)):
            image = Image(
                path_save,
                f'{file_names[i]}_vs_aoa',
                aoas_list,
                forces_list[i],
                labels=legends,
                label_x=labels_x[0],
                label_y=labels_y[i],
                size=18
            )
            colors = image.colors(len(aoas_list))
            image.plt_multicurve(colors,  legend_size=legend_size)


if __name__ == '__main__':
    path_root = r'F:\05_Special_Projects\202406_aircraft_project\cfd\03_ZQ10\simulation_fluent'
    path_comprehensive_post = os.path.join(path_root, 'comprehensive_post', '20250406')
    file = Files(path_root)
    file.create_folder('comprehensive_post')
    path_results = []
    labels = []
    for i in range(0, 100):
        labels.append(f'{(i+1):03d}')
        path = os.path.join(path_root, 'batch_results', f'ZQ10_wbt_20250406_{(i+1):03d}')
        path_results.append(path)
        aoas = np.linspace(-4, 20, 13)
        data_analysis = Fluent_Data_Analysis(path)
        data_analysis.write_excel(aoas, 'aoa')
        data_analysis.image_forces_vs_aoa()
    data_analysis_total = Fluent_Data_Analysis(path_results)
    data_analysis_total.image_forces_vs_aoa_comprehensive(path_comprehensive_post, labels, 2.0)
    data_analysis_total.write_excel_comprehensive(path_comprehensive_post)
