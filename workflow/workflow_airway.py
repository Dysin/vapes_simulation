'''
@Desc:   工作流
@Author: Dysin
@Date:   2025/11/20
'''

import csv
import sys

from workflow.workflow_base import WorkflowAirwayAnalysisBase
from geometry.geometry_utils import STLUtils
from utils.params_manager import *
from utils.files import Files,CSVUtils
from analysis.cfd_starccm.starccm_simulation import *
from analysis.cfd_starccm.starccm_data_analysis import StarCCMDataAnalysis
from analysis.post_tecplot.starccm_post import vape_post
from report.latex import ReportLatex
from uq.doe import DOE

class WorkflowRANS(WorkflowAirwayAnalysisBase):
    def __init__(
            self,
            product_model,
            version_number,
            mesh_folder='origin',
            mesh_user_name=None
    ):
        super().__init__(product_model, version_number, mesh_folder, mesh_user_name)

    def export_config(
            self,
            vape_type
    ):
        """
            将电子烟结构参数写入 CSVUtils 文件。
            参数:
                save_path (str): CSVUtils 文件保存路径（包含文件名，如 "D:/configure.csv"）
        """
        mesh_name = self.normalize_name()['mesh']
        path_config = os.path.join(self.pm.path_data, 'configure.csv')
        if not os.path.exists(path_config):
            geometry_utils = STLUtils(self.path_mesh)
            area_outlet = geometry_utils.get_area(f'{mesh_name}_outlet') * 1.0e6
            area_inlet = geometry_utils.get_area(f'{mesh_name}_inlet') * 1.0e6
            outlet_points = geometry_utils.get_points(f'{mesh_name}_outlet')
            ato_dir, cos_normal = geometry_utils.detect_plane_orientation(f'{mesh_name}_outlet')
            box_params = geometry_utils.get_bounding_box(f'{mesh_name}_atomizer_core')
            ato_diameter, ato_center = geometry_utils.get_cylinder_diameter(f'{mesh_name}_atomizer_core')

            ato_epsilon = 5.0e-5
            # 取方向轴 index
            axis_map = {'x': 'x', 'y': 'y', 'z': 'z'}
            axis = axis_map.get(ato_dir, 'y')  # 默认 y
            # 获取当前方向的 min/max
            min_v = box_params[f'min_{axis}']
            max_v = box_params[f'max_{axis}']
            outlet_coord = outlet_points[0][{'x': 0, 'y': 1, 'z': 2}[axis]]
            # 判断方向：outlet 在 min 的正侧 or 负侧
            positive = outlet_coord - min_v > 0
            if positive:
                ato_direction = axis
                ato_inlet_coord = (min_v + ato_epsilon) * 1.0e3
                ato_outlet_coord = (max_v - ato_epsilon) * 1.0e3
                ato_to_outlet_length = abs(outlet_coord - max_v) * 1.0e3
            else:
                ato_direction = f'-{axis}'
                ato_inlet_coord = (max_v - ato_epsilon) * 1.0e3
                ato_outlet_coord = (min_v + ato_epsilon) * 1.0e3
                ato_to_outlet_length = abs(outlet_coord - min_v) * 1.0e3
            # 公共部分
            ato_length = abs(max_v - min_v) * 1.0e3

            print(f'[INFO] 出气口面积：{area_outlet:.2f} mm^2')
            print(f'[INFO] 进气口面积：{area_inlet:.2f} mm^2')
            print(f'[INFO] 雾化芯方向: {ato_direction}')
            print(f'[INFO] 雾化区长度：{ato_length:.2f} mm')
            print(f'[INFO] 雾化区直径：{ato_diameter * 1.0e3:.2f} mm')
            print(f'[INFO] 雾化区入口位置：{ato_inlet_coord:.2f} mm')
            print(f'[INFO] 雾化区出口位置：{ato_outlet_coord:.2f} mm')
            print(f'[INFO] 雾化芯到出气口长度：{ato_to_outlet_length:.2f} mm')
            # 确保目录存在
            os.makedirs(os.path.dirname(path_config), exist_ok=True)
            # 要写入的数据
            data = [
                ['变量', '值'],
                ['进气口面积（mm^2）', f'{area_inlet:.2f}'],
                ['出气口面积（mm^2）', f'{area_outlet:.2f}'],
                ['雾化区长度（mm）', f'{ato_length:.2f}'],
                ['雾化区内径（mm）', f'{ato_diameter * 1.0e3:.2f}'],
                ['雾化芯到出气口长度（mm）', f'{ato_to_outlet_length:.2f}'],
                ['雾化区入口位置（mm）', f'{ato_inlet_coord:.2f}'],
                ['雾化区出口位置（mm）', f'{ato_outlet_coord:.2f}'],
                ['雾化芯方向', ato_direction],
                ['电子烟类型', vape_type]
            ]
            # 写入 CSVUtils
            with open(path_config, "w", newline='', encoding="utf-8-sig") as f:
                writer = csv.writer(f)
                writer.writerows(data)
            sys.exit(f'[ERROR] 文件不存在，已自动创建：{path_config}')

    def airway_simulation_and_post(
            self,
            flow_rate=17.5,
            mesh_name=None,
            bool_sim=True,
            bool_post=True,
            bool_res=False,
            report_folder=None
    ):
        '''
        气道仿真及后处理分析
        :return:
        '''
        get_params = GetParams()
        params_str = get_params.structure_config(self.path_proj)

        path_sim = os.path.join(self.pm.path_simulation, report_folder)
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
            files_mesh = Files(self.path_mesh)
            files_mesh.copy(f'{mesh_name}.nas', path_sim, 'airway.nas')
            starccm = StarCCMSimulation(path_sim)
            starccm.rans_single_phase_flow(
                part_name='airway',
                flow_rate=flow_rate,
                outlet_area=params_str.outlet_area,
                max_step=800,
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
        if bool_post:
            if bool_res:
                starccm_res = StarCCMDataAnalysis(path_sim)
                starccm_res.plt_curve('pressure_ave_inlet')
                starccm_res.plt_curve('pressure_ave_outlet')
                starccm_res.plt_curve('pressure_ave_airflow_sensor')
            path_report = os.path.join(self.pm.path_reports, str(report_folder))
            vape_post(
                path_geo=self.path_mesh,
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
                path_root=self.path_proj,
                folder_name=report_folder,
                model_number=self.product_model,
                params_structure=params_str,
                flow_rates=flow_rate,
                version_number=self.ver_num
            )
            report_latex.convert_images()
            report_latex.rans_spf_single(report_folder)

    def rans_spf_default_flow_rates(self, vape_type):
        self.export_config(vape_type)
        get_params = GetParams()
        params_str = get_params.structure_config(self.path_proj)
        basic_flow_rates = [17.5, 20.5, 22.5]
        for i in range(0, len(basic_flow_rates)):
            mesh_name, case_name = self.normalize_name(
                basic_flow_rates[i],
                self.mesh_user_name
            )
            self.airway_simulation_and_post(
                flow_rate=basic_flow_rates[i],
                mesh_name=mesh_name,
                bool_sim=True,
                bool_post=True,
                bool_res=False,
                report_folder=case_name
            )

        if self.mesh_user_name is None:
            report_folder = f'{self.ver_num}_rans_spf_flowrate_compare'
        else:
            report_folder = f'{self.ver_num}_{self.mesh_user_name}_rans_spf_flowrate_compare'
        path_report = os.path.join(self.pm.path_reports, report_folder)
        files_images = Files(self.pm.path_images)
        geo_imgs = files_images.get_file_names_by_type('.png')
        for geo_img in geo_imgs:
            files_images.copy(
                f'{geo_img}.png',
                os.path.join(path_report, 'images')
            )
        report_latex = ReportLatex(
            path_root=self.path_proj,
            folder_name=report_folder,
            model_number=self.product_model,
            params_structure=params_str,
            flow_rates=basic_flow_rates,
            version_number=self.ver_num
        )
        report_latex.rans_spf_comparison_flow_rates(user_name=self.mesh_user_name)

    def rans_spf_user_flow_rates(
            self,
            vape_type,
            flow_rates,
            star_num=0,
            bool_sim=True,
            bool_post=False
    ):
        self.export_config(vape_type)
        for i in range(star_num, len(flow_rates)):
            mesh_name, case_name = self.normalize_name(flow_rates[i], self.mesh_user_name)
            self.airway_simulation_and_post(
                flow_rate=flow_rates[i],
                mesh_name=mesh_name,
                bool_sim=bool_sim,
                bool_post=bool_post,
                bool_res=False,
                report_folder=case_name
            )

    def rans_spf_experiment_compare(
            self,
            flow_rates=None
    ):
        get_params = GetParams()
        params_str = get_params.structure_config(self.path_proj)
        basic_flow_rates = [17.5, 20.5, 22.5]
        for i in range(0, len(basic_flow_rates)):
            mesh_name, case_name = self.normalize_name(
                basic_flow_rates[i],
                self.mesh_user_name
            )
            self.airway_simulation_and_post(
                flow_rate=basic_flow_rates[i],
                mesh_name=mesh_name,
                bool_sim=False,
                bool_post=True,
                bool_res=True,
                report_folder=case_name
            )

        if self.mesh_user_name is None:
            report_folder = f'{self.ver_num}_rans_spf_experiment_compare'
        else:
            report_folder = f'{self.ver_num}_{self.mesh_user_name}_rans_spf_experiment_compare'
        path_report = os.path.join(self.pm.path_reports, report_folder)
        files_images = Files(self.pm.path_images)
        geo_imgs = files_images.get_file_names_by_type('.png')
        for geo_img in geo_imgs:
            files_images.copy(
                f'{geo_img}.png',
                os.path.join(path_report, 'images')
            )
        report_latex = ReportLatex(
            path_root=self.path_proj,
            folder_name=report_folder,
            model_number=self.product_model,
            params_structure=params_str,
            flow_rates=basic_flow_rates,
            version_number=self.ver_num
        )
        report_latex.rans_spf_comparison_flow_rates(
            user_name=self.mesh_user_name,
            bool_exp=True,
            all_flow_rates=flow_rates
        )

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
        params_str = get_params.structure_config(self.path_proj)
        basic_flow_rates = [17.5, 20.5, 22.5]
        if mesh_user_name is None:
            report_folder = f'{self.ver_num}_rans_spf_optimization_compare'
        else:
            report_folder = f'{self.ver_num}_{mesh_user_name}_rans_spf_optimization_scheme'
        report_latex = ReportLatex(
            path_root=self.path_proj,
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

    def rans_spf_user_report(
            self,
            report_folder,
            flow_rate,
            case_names,
            col_names,
            captions
    ):
        path_report = os.path.join(self.pm.path_reports, str(report_folder))
        files_images = Files(self.pm.path_images)
        geo_imgs = files_images.get_file_names_by_type('.png')
        for geo_img in geo_imgs:
            files_images.copy(
                f'{geo_img}.png',
                os.path.join(path_report, 'images')
            )
        get_params = GetParams()
        params_str = get_params.structure_config(self.path_proj)
        report_latex = ReportLatex(
            path_root=self.path_proj,
            folder_name=report_folder,
            model_number=self.product_model,
            params_structure=params_str,
            flow_rates=flow_rate,
            version_number=self.ver_num
        )
        report_latex.rans_spf_comparison_user(
            case_names,
            col_names,
            captions
        )

    def spf_uq(self, region_num):
        self.get_airway_opt_region_params(region_num)

if __name__ == '__main__':
    vape_name = 'VP353'
    ver_num = '20251201'
    # mesh_user_name = 'modify_r0p4'

    vape_type = '一次性'
    workflow = WorkflowRANS(vape_name, ver_num, mesh_folder='doe')
    # workflow.rans_spf_optimization(22.5)

    # flow_rates = np.arange(17.5, 30, 2.5).tolist()
    # print(flow_rates)
    # workflow.rans_spf_user_flow_rates(
    #     vape_type,
    #     [40.0],
    #     mesh_user_name=mesh_user_name,
    #     bool_sim=True,
    #     bool_post=True
    # )
    # workflow.rans_spf_user_report(
    #     report_folder=f'20251201_modify_rans_spf_optimization_inlets',
    #     flow_rate=40.0,
    #     case_names=[
    #         '20251201_modify_rans_spf_q40.0',
    #         '20251201_modify_r0p2_rans_spf_q40.0',
    #         '20251201_modify_r0p4_rans_spf_q40.0'
    #     ],
    #     col_names=[
    #         '原气道', '进气口R0.2', '进气口R0.4'
    #     ],
    #     captions=[
    #         'R0.0mm', 'R0.2mm', 'R0.4mm'
    #     ]
    # )
    workflow.spf_uq(2)