'''
@Desc:   工作流父类
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

class WorkflowAirwayAnalysisBase:
    def __init__(
            self,
            product_model,
            version_number,
            mesh_folder,
            mesh_user_name,
            path_root=f'E:\\1_Work\\active'
    ):
        '''
        气道内流场计算工作流，基类
        :param product_model: 产品型号
        :param version_number: 版本号
        :param mesh_folder: 网格文件夹名
        :param path_root: 根目录
        '''
        self.product_model = product_model # 产品型号
        self.proj_name = product_model.replace('-', '') # 项目名
        self.ver_num = version_number # 版本号，如20251010
        self.path_root = path_root
        self.path_airway = os.path.join(self.path_root, 'airway_analysis')
        self.path_proj = os.path.join(self.path_airway, self.proj_name)
        self.pm = PathManager(self.path_proj, True)
        self.mesh_user_name = mesh_user_name
        self.path_mesh = os.path.join(self.pm.path_mesh, str(mesh_folder))
        self.check_exist(self.path_mesh)
    def check_exist(self, path):
        '''
        检测路径或文件是否存在，若不存在则创建并终止程序输出报错信息
        :param path: 路径
        :return:
        '''
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
            sys.exit(f"[ERROR] 路径不存在，已自动创建：{path}")

    def normalize_name(
            self,
            flow_rate: float = None,
            batch_num: int = None,
            opt_num: int = None,
    ):
        '''
        统一命名生成器（始终返回全部名称）
        1.网格命名规范：{version_number}{_{user_name}}_airway
        2.算例文件夹命名规范：
            {version_number}{_{user_name}}_rans_spf_q{flow_rate}_b{batch_num}_p{opt_num}
            {version_number}{_{user_name}}_ddes_spf_q{flow_rate}_b{batch_num}_p{opt_num}
            {version_number}{_{user_name}}_rans_mpf_q{flow_rate}_b{batch_num}_p{opt_num}
            {version_number}{_{user_name}}_ddes_mpf_q{flow_rate}_b{batch_num}_p{opt_num}
        返回:
            {
                "mesh": "...",
                "rans_spf": "...",
                "ddes_spf": "...",
                "rans_mpf": "...",
                "ddes_mpf": "...",
            }
        '''
        # 基础前缀
        parts = [self.ver_num]
        if self.mesh_user_name is not None:
            parts.append(self.mesh_user_name)

        prefix = "_".join(parts)

        # batch / opt 后缀
        if batch_num is None:
            suffix = ""
            mesh_name = f"{prefix}_airway"
        else:
            suffix = f"_b{batch_num:02d}_p{opt_num:02d}"
            mesh_name = f"{prefix}_airway{suffix}"
        # 工具函数：构造 case 名
        def make_case(model):
            return f"{prefix}_{model}_q{flow_rate}{suffix}"
        # 全部名称统一返回
        names = {
            "mesh": mesh_name,
            "rans_spf": make_case("rans_spf"),
            "ddes_spf": make_case("ddes_spf"),
            "rans_mpf": make_case("rans_mpf"),
            "ddes_mpf": make_case("ddes_mpf"),
        }
        return names

    def get_airway_opt_region_params(self, region_num):
        '''
        获取气道优化区域（一般是圆柱STL）的参数，如圆心坐标，直径等
        :param region_num: 区域数
        :return:
        '''
        mesh_name = self.normalize_name()['mesh']
        csv_region_params = os.path.join(self.path_mesh, 'opt_region_params.csv')
        data = [
            ['region_name', 'center1', 'center2', 'radius', 'axis']
        ]
        for i in range(region_num):
            region_name = f'{mesh_name}_opt_region{i+1}'
            geo_utils = STLUtils(self.path_mesh, region_name)
            cyl_params = geo_utils.get_cylinder_params()
            row = [
                region_name,
                cyl_params['center1'],
                cyl_params['center2'],
                cyl_params['radius'],
                cyl_params['axis']
            ]
            data.append(row)
        with open(csv_region_params, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerows(data)