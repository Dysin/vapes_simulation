'''
@Desc:   工作流
@Author: Dysin
@Date:   2025/11/20
'''

import csv
import sys
import numpy as np
from utils.params_manager import *
from utils.files import Files
from analysis.cfd_starccm.starccm_simulation import *
from analysis.cfd_starccm.starccm_data_analysis import StarCCMDataAnalysis
from analysis.cfd_tecplot.starccm_post import vape_post
from report.latex import ReportLatex

class Workflow:
    def __init__(self, product_model, version_number):
        self.product_model = product_model # 产品型号
        self.proj_name = product_model.replace('-', '') # 项目名
        self.ver_num = version_number # 版本号，如20251010
        self.path_root = f'E:\\1_Work\\active'
        self.path_airway = os.path.join(self.path_root, 'airway_analysis')
        self.path_airway_proj = os.path.join(self.path_airway, self.proj_name)
        self.pm = PathManager(self.path_airway_proj, True)

    def check_exist(self, path):
        '''
        检测路径或文件是否存在，若不存在则创建并终止程序输出报错信息
        :param path: 路径
        :return:
        '''
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
            sys.exit(f"[ERROR] 路径不存在，已自动创建：{path}")

    def export_config(self):
        """
            将电子烟结构参数写入 CSVUtils 文件。
            参数:
                save_path (str): CSVUtils 文件保存路径（包含文件名，如 "D:/configure.csv"）
        """
        path_config = os.path.join(self.pm.path_data, 'configure.csv')
        if not os.path.exists(path_config):
            # 确保目录存在
            os.makedirs(os.path.dirname(path_config), exist_ok=True)
            # 要写入的数据
            data = [
                ["变量", "值"],
                ["进气口面积（mm^2）", 0],
                ["出气口面积（mm^2）", 0],
                ["雾化区长度（mm）", 0],
                ["雾化区内径（mm）", 0],
                ["雾化芯到出气口长度（mm）", 0],
                ["雾化区入口位置（mm）", 0],
                ["雾化区出口位置（mm）", 0],
                ["雾化芯方向", "y"],
                ["电子烟类型", "一次性"]
            ]
            # 写入 CSVUtils
            with open(path_config, "w", newline='', encoding="utf-8-sig") as f:
                writer = csv.writer(f)
                writer.writerows(data)
            sys.exit(f'[ERROR] 文件不存在，已自动创建：{path_config}')

    def naming_conventions(self, flow_rate, user_name):
        '''
        命名规范
        1.网格命名规范：{version_number}{_{user_name}}_airway
        2.算例文件夹命名规范：{version_number}{_{user_name}}_rans_spf_q{flow_rate}
                          {version_number}{_{user_name}}_ddes_spf_q{flow_rate}
                          {version_number}{_{user_name}}_rans_mpf_q{flow_rate}
                          {version_number}{_{user_name}}_ddes_mpf_q{flow_rate}
        :param flow_rate: 体积流率
        :return: 网格名&算例名
        '''
        if user_name is None:
            mesh_name = f'{self.ver_num}_airway'
            case_name = f'{self.ver_num}_rans_spf_q{flow_rate}'
        else:
            mesh_name = f'{self.ver_num}_{user_name}_airway'
            case_name = f'{self.ver_num}_{user_name}_rans_spf_q{flow_rate}'
        return mesh_name, case_name

    def airway_simulation_and_post(
            self,
            flow_rate=17.5,
            mesh_folder='origin',
            user_name=None,
            bool_sim=True,
            bool_post=True,
            bool_res=False,
            report_folder=None
    ):
        '''
        气道仿真及后处理分析
        :return:
        '''
        path_mesh = os.path.join(self.pm.path_mesh, mesh_folder)
        self.check_exist(path_mesh)
        get_params = GetParams()
        params_str = get_params.structure_config(self.path_airway_proj)

        mesh_name, case_name = self.naming_conventions(flow_rate, user_name)
        path_sim = os.path.join(self.pm.path_simulation, case_name)
        # 检查images路径中是否存在图片
        files_images = Files(self.pm.path_images)
        geo_imgs = files_images.get_file_names_by_type('.png')
        geos_check = ['geometry_inlet', 'geometry_airway']
        for geo_check in geos_check:
            if geo_check not in geo_imgs:
                sys.exit(f'[WARNING] {self.pm.path_images} 不存在 {geo_check}.png')
        if bool_sim:
            target_size = 2e-4
            min_size = 8e-5
            max_size = 2 * target_size
            num_layers = 10
            files_mesh = Files(path_mesh)
            files_mesh.copy(f'{mesh_name}.nas', path_sim, 'airway.nas')
            starccm = StarCCMSimulation(path_sim)
            starccm.rans_single_phase_flow(
                part_name='airway',
                flow_rate=flow_rate,
                outlet_area=params_str.outlet_area,
                max_step=1000,
                bool_gen_mesh=True,
                target_mesh_size=target_size,
                min_mesh_size=min_size,
                max_mesh_size=max_size,
                num_layers=num_layers,
                atomization_area_pos=[
                    params_str.ato_inlet_coord,
                    params_str.ato_outlet_coord
                ],
                atomization_area_dir=params_str.ato_direction
            )
            starccm.run(new=True)
            if bool_res:
                starccm_res = StarCCMDataAnalysis(path_sim)
                starccm_res.plt_curve('pressure_ave_inlet')
                starccm_res.plt_curve('pressure_ave_outlet')
                starccm_res.plt_curve('pressure_ave_airflow_sensor')
        if bool_post:
            path_report = os.path.join(self.pm.path_reports, case_name)
            vape_post(
                path_geo=path_mesh,
                path_post=path_sim,
                geo_name=mesh_name,
                file_name='airway',
                atomization_area_pos=[
                    params_str.ato_inlet_coord,
                    params_str.ato_outlet_coord
                ],
                atomization_area_dir=params_str.ato_direction
            )

            # 输出报告
            for geo_img in geo_imgs:
                files_images.copy(
                    f'{geo_img}.png',
                    os.path.join(path_report, 'images')
                )
            report_latex = ReportLatex(
                path_root=self.path_airway_proj,
                folder_name=report_folder,
                model_number=self.product_model,
                params_structure=params_str,
                flow_rates=flow_rate,
                version_number=self.ver_num
            )
            report_latex.convert_images()
            report_latex.rans_spf_single(case_name)

    def rans_spf_basic_flow_rates(self, mesh_user_name=None):
        self.export_config()
        get_params = GetParams()
        params_str = get_params.structure_config(self.path_airway_proj)
        # basic_flow_rates = [17.5, 20.5, 22.5]
        basic_flow_rates = [30.5, 35.5, 40.5]
        for i in range(0, len(basic_flow_rates)):
            mesh_name, case_name = self.naming_conventions(
                basic_flow_rates[i],
                mesh_user_name
            )
            self.airway_simulation_and_post(
                flow_rate=basic_flow_rates[i],
                mesh_folder='origin',
                user_name=mesh_user_name,
                bool_sim=True,
                bool_post=True,
                bool_res=False,
                report_folder=case_name
            )

        if mesh_user_name is None:
            report_floder = f'{self.ver_num}_rans_spf_flowrate_compare'
        else:
            report_floder = f'{self.ver_num}_{mesh_user_name}_rans_spf_flowrate_compare'
        path_report = os.path.join(self.pm.path_reports, report_floder)
        files_images = Files(self.pm.path_images)
        geo_imgs = files_images.get_file_names_by_type('.png')
        for geo_img in geo_imgs:
            files_images.copy(
                f'{geo_img}.png',
                os.path.join(path_report, 'images')
            )
        report_latex = ReportLatex(
            path_root=self.path_airway_proj,
            folder_name=report_floder,
            model_number=self.product_model,
            params_structure=params_str,
            flow_rates=basic_flow_rates,
            version_number=self.ver_num
        )
        report_latex.rans_spf_comparison_flow_rates(user_name=mesh_user_name)

    def rans_spf_user_flow_rates(
            self,
            flow_rates,
            star_num=0,
            mesh_user_name=None
    ):
        self.export_config()
        for i in range(star_num, len(flow_rates)):
            mesh_name, case_name = self.naming_conventions(
                flow_rates[i],
                mesh_user_name
            )
            self.airway_simulation_and_post(
                flow_rate=flow_rates[i],
                mesh_folder='origin',
                user_name=mesh_user_name,
                bool_sim=True,
                bool_post=False,
                bool_res=False,
                report_folder=case_name
            )

if __name__ == '__main__':
    vape_name = 'ATN-021'
    ver_num = '20251015'
    workflow = Workflow(vape_name, ver_num)
    # workflow.rans_spf_basic_flow_rates()

    flow_rates = np.arange(17.5, 30, 2.5).tolist()
    print(flow_rates)
    # workflow.rans_spf_user_flow_rates(flow_rates)