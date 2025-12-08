'''
@Desc:   不确定性量化工作流
@Author: Dysin
@Date:   2025/12/4
'''

import csv
import sys
import numpy as np

from geometry.geometry_utils import STLUtils
from utils.params_manager import *
from utils.files_utils import FileUtils,CSVUtils
from analysis.cfd_starccm.starccm_simulation import *
from analysis.cfd_starccm.starccm_data_analysis import StarCCMDataAnalysis
from analysis.post_tecplot.starccm_post import vape_post
from report.latex import ReportLatex
from uq.doe import DOE

class WorkflowUQ:
    def __init__(self, path_data):
        self.path_data = path_data

    def airway_design_doe(self, input_param_ranges, sample_num, ver_num, csv_id):
        doe = DOE(input_param_ranges, sample_num)
        data = doe.latin_hypercube_sampling()
        path_input = os.path.join(
            self.path_data,
            f'{ver_num}_input_params_{csv_id:02d}.csv'
        )
        # 生成列标题
        headers = [f'region{i + 1}' for i in range(data.shape[1])]
        # 写入CSV文件
        with open(path_input, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            # 写入标题行
            writer.writerow(headers)
            # 写入数据行
            writer.writerows(data)
        print(f"数据已成功写入 {path_input}")

    def get_airway_opt_region_params(self, region_num):
        '''
        获取气道优化区域（一般是圆柱STL）的参数，如圆心坐标，直径等
        :param region_num: 区域数
        :return:
        '''
        path_doe = os.path.join()
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
            sys.exit(f"[ERROR] 路径不存在，已自动创建：{path}")


    def rans_spf_optimization(self, flow_rate):
        csv_id = 1
        csv_manager = CSVUtils(
            self.pm.path_data,
            f'{ver_num}_input_params_{csv_id:02d}'
        )
        df_input = csv_manager.read()
        # for i in range(0, len(df_input)):
        #     mesh_name, case_name = self.naming_conventions(
        #         flow_rate,
        #         mesh_user_name,
        #         batch_num=csv_id,
        #         opt_num=i+1
        #     )
        #     self.airway_simulation_and_post(
        #         flow_rate=flow_rate,
        #         mesh_name=mesh_name,
        #         bool_sim=True,
        #         bool_post=False,
        #         bool_res=False,
        #         report_folder=case_name
        #     )
        get_params = GetParams()
        params_str = get_params.structure_config(self.path_airway_proj)
        basic_flow_rates = [17.5, 20.5, 22.5]
        if mesh_user_name is None:
            report_folder = f'{self.ver_num}_rans_spf_optimization_compare'
        else:
            report_folder = f'{self.ver_num}_{mesh_user_name}_rans_spf_optimization_scheme'
        report_latex = ReportLatex(
            path_root=self.path_airway_proj,
            folder_name=report_folder,
            model_number=self.product_model,
            params_structure=params_str,
            flow_rates=basic_flow_rates,
            version_number=self.ver_num
        )
        report_latex.rans_spf_comparison_optimization(
            csv_id,
            None,
            None
        )