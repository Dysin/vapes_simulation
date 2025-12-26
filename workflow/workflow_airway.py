'''
@Desc:   工作流
@Author: Dysin
@Date:   2025/11/20
'''

import csv
import os.path
import sys

import pandas as pd
from workflow.workflow_base import WorkflowAirwayAnalysisBase
from geometry.geometry_utils import STLUtils
from utils.params_manager import *
from utils.files_utils import FileUtils,CSVUtils
from analysis.cfd_starccm.starccm_simulation import *
from analysis.cfd_starccm.starccm_data_analysis import StarCCMDataAnalysis
from analysis.post_tecplot.starccm_post import vape_post
from analysis.mesh_ansa.ansa_run import run_ansa
from report.latex import ReportLatex
from uq.doe import DOE
from uq.surrogate_model import SurrogateModelBasic
from utils.images_utils import PlotImage3D
from uq.gp_analysis import GPAnalyzer

class WorkflowRANS(WorkflowAirwayAnalysisBase):
    def __init__(
            self,
            product_model,
            version_number,
            mesh_folder='origin',
            mesh_user_name=None
    ):
        super().__init__(product_model, version_number, mesh_folder, mesh_user_name)
        self.path_source_ansa = r'E:\1_Work\templates\vapes_simulation\source\analysis\mesh_ansa'
        self.path_ansa = r'C:\Users\HG\AppData\Local\Apps\BETA_CAE_Systems\ansa_v19.1.1'

    def export_config(
            self,
            vape_type,
            bool_new = True
    ):
        """
            将电子烟结构参数写入 CSVUtils 文件。
            参数:
                save_path (str): CSVUtils 文件保存路径（包含文件名，如 "D:/configure.csv"）
        """
        name = self.normalize_name()
        mesh_name = name['mesh']
        path_config = os.path.join(self.pm.path_data, 'configure.csv')
        if not os.path.exists(path_config) or bool_new:
            geometry_utils = STLUtils(self.path_mesh, f'{mesh_name}_outlet')
            area_outlet = geometry_utils.get_area() * 1.0e6
            geometry_utils = STLUtils(self.path_mesh, f'{mesh_name}_inlet')
            area_inlet = geometry_utils.get_area() * 1.0e6
            geometry_utils = STLUtils(self.path_mesh, f'{mesh_name}_outlet')
            outlet_points = geometry_utils.get_points()
            geometry_utils = STLUtils(self.path_mesh, f'{mesh_name}_outlet')
            ato_dir, cos_normal = geometry_utils.detect_plane_orientation()
            geometry_utils = STLUtils(self.path_mesh, f'{mesh_name}_atomizer_core')
            box_params = geometry_utils.get_bounding_box()
            geometry_utils = STLUtils(self.path_mesh, f'{mesh_name}_atomizer_core')
            ato_diameter, ato_center = geometry_utils.get_cylinder_diameter()

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
            # sys.exit(f'[ERROR] 文件不存在，已自动创建：{path_config}')

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
        files_images = FileUtils(self.pm.path_images)
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
            files_mesh = FileUtils(self.path_mesh)
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

    def spf_default_flow_rates(self, vape_type):
        # 输出mesh
        if self.mesh_user_name is not None:
            mesh_name = f'{self.ver_num}_{self.mesh_user_name}_airway'
        else:
            mesh_name = f'{self.ver_num}_airway'
        output_script = os.path.join(self.path_source_ansa, 'output_mesh.py')
        file_utils_output = FileUtils(output_script)
        txt = {
            62: f'    mesh_name = \'{mesh_name}\'',
            63: f'    path_mesh = r\'{self.path_mesh}\'',
            64: f'    output_mesh(path_mesh, mesh_name)',
        }
        file_utils_output.modify_multiple_lines(txt, True)
        run_ansa(self.path_ansa, output_script, bool_gui=False)
        self.export_config(vape_type)
        get_params = GetParams()
        params_str = get_params.structure_config(self.path_proj)
        basic_flow_rates = [17.5, 20.5, 22.5]
        for i in range(0, len(basic_flow_rates)):
            names = self.normalize_name(
                basic_flow_rates[i]
            )
            self.airway_simulation_and_post(
                flow_rate=basic_flow_rates[i],
                mesh_name=names['mesh'],
                bool_sim=True,
                bool_post=True,
                bool_res=False,
                report_folder=names['rans_spf']
            )

        if self.mesh_user_name is None:
            report_folder = f'{self.ver_num}_rans_spf_flowrate_compare'
        else:
            report_folder = f'{self.ver_num}_{self.mesh_user_name}_rans_spf_flowrate_compare'
        path_report = os.path.join(self.pm.path_reports, report_folder)
        files_images = FileUtils(self.pm.path_images)
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

    def spf_user_flow_rates(
            self,
            vape_type,
            flow_rates,
            star_num=0,
            bool_sim=True,
            bool_post=False
    ):
        # 输出mesh
        if self.mesh_user_name is not None:
            mesh_name = f'{self.ver_num}_{self.mesh_user_name}_airway'
        else:
            mesh_name = f'{self.ver_num}_airway'
        output_script = os.path.join(self.path_source_ansa, 'output_mesh.py')
        file_utils_output = FileUtils(output_script)
        txt = {
            62: f'    mesh_name = \'{mesh_name}\'',
            63: f'    path_mesh = r\'{self.path_mesh}\'',
            64: f'    output_mesh(path_mesh, mesh_name)',
        }
        file_utils_output.modify_multiple_lines(txt, True)
        run_ansa(self.path_ansa, output_script, bool_gui=False)
        self.export_config(vape_type)
        for i in range(star_num, len(flow_rates)):
            names = self.normalize_name(
                flow_rates[i]
            )
            self.airway_simulation_and_post(
                flow_rate=flow_rates[i],
                mesh_name=names['mesh'],
                bool_sim=bool_sim,
                bool_post=bool_post,
                bool_res=False,
                report_folder=names['rans_spf']
            )

    def spf_experiment_compare(
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
        files_images = FileUtils(self.pm.path_images)
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

    def spf_user_report(
            self,
            report_folder,
            flow_rate,
            case_names,
            col_names,
            captions
    ):
        path_report = os.path.join(self.pm.path_reports, str(report_folder))
        files_images = FileUtils(self.pm.path_images)
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

    def get_csv_params(self, flow_rate):
        # 合并输入参数
        file_utils = FileUtils(self.pm.path_data)
        csv_input_name = f'{self.ver_num}_input_params'
        file_utils.remove(f'{csv_input_name}.csv')
        csv_input_names = file_utils.filter_star_filenames(
            csv_input_name,
            None
        )
        batch_num = len(csv_input_names)
        dfs = []
        opt_nums = []
        for i in range(batch_num):
            file_csv = os.path.join(
                self.pm.path_data,
                str(csv_input_names[i])
            )
            opt_nums.append(len(pd.read_csv(file_csv, header=0)))
            if i == 0:
                dfs.append(pd.read_csv(file_csv))
            else:
                dfs.append(pd.read_csv(file_csv, header=0))
        df_input = pd.concat(dfs, ignore_index=True)
        file_csv_input = os.path.join(self.pm.path_data, f'{csv_input_name}.csv')
        df_input.to_csv(file_csv_input, index=False)
        case_names = []
        for i in range(batch_num):
            for j in range(opt_nums[i]):
                names = self.normalize_name(
                    flow_rate,
                    batch_num=i + 1,
                    opt_num=j + 1
                )
                case_names.append(names['rans_spf'])
        csv_utils = CSVUtils(
            self.pm.path_data,
            f'{self.ver_num}_output_params'
        )
        csv_utils.remove()
        row_headers_output = [
            'delta_p_flow',
            'delta_p_sensor',
            'curle_acoustic_power_max',
            'ti_ave_outlet'
        ]
        csv_utils.write_row_data(row_headers_output)
        for i in range(len(case_names)):
            path_sim = os.path.join(self.pm.path_simulation, case_names[i])
            params_sim = GetParams().simulation(path_sim)
            row = [
                params_sim.delta_p_flow,
                params_sim.delta_p_sensor,
                params_sim.curle_acoustic_power_max,
                params_sim.ti_ave_outlet
            ]
            csv_utils.write_row_data(row)
        df_output = csv_utils.read()
        return df_input, df_output

    def spf_uq(
            self,
            bool_doe=True,
            bool_morph=True,
            bool_sim=True,
            bool_latex=True,
            bool_uq=True,
            input_info=None,
            batch_id=1,
            sample_num=10,
            flow_rate=None,
            sim_star_num=1
    ):
        r2 = 0
        case_names = []
        var_names = []  # 变量名
        var_ranges = [] # 变量上下界
        for key, value in input_info.items():
            var_names.append(key)
            var_ranges.append(value)
        while r2 < 0.5 and batch_id < 10:
            region_num = len(var_ranges)
            if bool_doe:
                doe = DOE(var_ranges, sample_num)
                data_input = doe.latin_hypercube_sampling()
                csv_input = os.path.join(
                    self.pm.path_data,
                    f'{self.ver_num}_input_params_{batch_id:02d}.csv'
                )
                header = [f'region{i + 1}' for i in range(region_num)]
                with open(csv_input, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(header)
                    writer.writerows(data_input)
            if bool_morph:
                output_script = os.path.join(self.path_source_ansa, 'output_mesh_opt_region.py')
                file_utils_output = FileUtils(output_script)
                txt = {
                    35: f'    mesh_name = \'{self.ver_num}_airway\'',
                    36: f'    path_mesh = r\'{self.path_mesh}\'',
                    37: f'    output_opt_region_mesh(path_mesh, mesh_name, {region_num})',
                }
                file_utils_output.modify_multiple_lines(txt, True)
                run_ansa(self.path_ansa, output_script, bool_gui=False)

                self.get_airway_opt_region_params(region_num)

                morph_script = os.path.join(self.path_source_ansa, 'morph.py')
                file_utils_morph = FileUtils(morph_script)
                txt = {
                    128: f'    proj_name = \'{self.proj_name}\'',
                    129: f'    ver_number = \'{self.ver_num}\'',
                    130: f'    batch_id = {batch_id}',
                }
                file_utils_morph.modify_multiple_lines(txt, True)
                run_ansa(self.path_ansa, morph_script, bool_gui=True)
            if bool_sim:
                for i in range(sim_star_num - 1, sample_num):
                    names = self.normalize_name(
                        flow_rate,
                        batch_num=batch_id,
                        opt_num=i+1
                    )
                    case_names.append(names['rans_spf'])
                    # print(names['mesh'], names['rans_spf'])
                    self.airway_simulation_and_post(
                        flow_rate=flow_rate,
                        mesh_name=names['mesh'],
                        bool_sim=True,
                        bool_post=False,
                        bool_res=False,
                        report_folder=names['rans_spf']
                    )

            if bool_uq:
                # 代理模型
                df_input, df_output = self.get_csv_params(flow_rate)
                gp_analyzer = GPAnalyzer(df_input, df_output)
                gp_analyzer.workflow(
                    path_image=self.pm.path_images,
                    problem_params=input_info
                )
            batch_id += 1
            sys.exit()

        if bool_latex:
            get_params = GetParams()
            params_str = get_params.structure_config(self.path_proj)
            if self.mesh_user_name is None:
                report_folder = f'{self.ver_num}_rans_spf_optimization_scheme'
            else:
                report_folder = f'{self.ver_num}_{self.mesh_user_name}_rans_spf_optimization_scheme'
            report_latex = ReportLatex(
                path_root=self.path_proj,
                folder_name=report_folder,
                model_number=self.product_model,
                params_structure=params_str,
                flow_rates=flow_rate,
                version_number=self.ver_num
            )
            report_latex.rans_spf_comparison_optimization(
                batch_id,
                case_names,
                None,
                None
            )

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
    # input_params_range = [
    #     [-0.4, -0.1],
    #     [-0.4, -0.1],
    #     [-0.2, -0.1],
    #     [-0.2, -0.1]
    # ] # 2025.12.09 am

    input_params_info = {
        'Region1 delta R': (-0.2, -0.1),
        'Region2 delta R': (0.1, 0.15),
        'Region3 delta R': (-0.1, 0.1),
    }
    workflow.spf_uq(
        bool_doe=False,
        bool_morph=False,
        bool_sim=False,
        bool_latex=False,
        input_info=input_params_info,
        batch_id=5,
        sample_num=10,
        flow_rate=22.5,
        sim_star_num=9
    )