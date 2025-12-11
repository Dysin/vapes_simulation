'''
@Desc:   
@Author: Dysin
@Date:   2025/10/14
'''

from report.latex_utils import *
from utils.files_utils import FileUtils
from utils.images import ImageUtils
from report.results_analysis import ResultsAnalysis
from utils.global_variables import *
from utils.params_manager import *
from utils.files_utils import Image

class ReportLatex(object):
    def __init__(
            self,
            path_root,
            folder_name: str,
            model_number: str,
            params_structure: StructureParams,
            flow_rates,
            version_number=None
    ):
        self.pm = PathManager(path_root, False)
        self.folder_name = folder_name
        self.vape_name = model_number
        self.proj_name = model_number.replace('-', '')
        self.params_str = params_structure
        self.path_latex = os.path.join(
            self.pm.path_reports,
            self.folder_name,
        )
        self.path_latex_images = os.path.join(self.path_latex, 'images')
        os.makedirs(self.path_latex_images, exist_ok=True)
        self.latex = LatexUtils(
            vape_name=self.vape_name,
            path=self.path_latex
        )
        self.files_images = FileUtils(self.path_latex_images)
        self.flow_rates = flow_rates
        self.results_analysis = ResultsAnalysis()
        self.version_number = version_number

    def convert_images(self):
        path_simulation = os.path.join(self.pm.path_simulation, self.folder_name)
        files_images = FileUtils(path_simulation)
        image_manager = ImageUtils()
        image_names = files_images.get_file_names_by_type('.bmp')
        for image_name in image_names:
            image_manager.bmp_to_png(
                path_simulation,
                self.path_latex_images,
                image_name,
                add_border=True
            )

    def insert_figures_single_sim(self, txt, search_string, main_captions=None):
        '''
        插入图片，单个算例
        :return:
        '''

        image_names = self.files_images.filter_star_filenames(
            search_string=search_string,
            suffix_to_remove='.png'
        )
        print(image_names)
        if main_captions is None:
            main_captions = self.latex.pair_caption_list(image_names)
        res = (f'{self.vape_name}不同{txt}如下图所示：\n')
        res += self.latex.insert_figures_two_per_row(
            image_names,
            captions=None,
            main_captions=main_captions
        )
        return res

    def insert_figures_diff_sim(
            self,
            txt,
            search_string,
            folder_names,
            row_captions,
            main_captions=None
    ):
        """
        生成 LaTeX 多行多列图片代码：
        每行表示一个变量/方向位置，不同列对应不同流率。
        参数：
            txt: str                  插图前描述文字
            search_string: str        搜索关键字，用于筛选图片文件名
        """
        # === 获取所有文件夹下的图片 ===
        image_names_all = []  # 每个流率下的图片名列表（列方向）
        for folder_name in folder_names:
            path_images = os.path.join(self.pm.path_reports, folder_name, 'images')
            files_images = FileUtils(path_images)
            image_names = files_images.filter_star_filenames(
                search_string=search_string,
                suffix_to_remove='.png'
            )
            # 加相对路径：../{folder_name}/images/{image_name}
            image_names = [f'../{folder_name}/images/{img}' for img in image_names]
            image_names_all.append(image_names)
        # === 转置，使得结构变为 n 行 × m 列 ===
        # n 为同类型图片数量，m 为流率数量
        image_names_matrix = list(map(list, zip(*image_names_all)))
        # === 读取第一个流率的图片名，用于生成标题 ===
        path_first = os.path.join(self.pm.path_reports, folder_names[0], 'images')
        files_first = FileUtils(path_first)
        image_names_first = files_first.filter_star_filenames(
            search_string=search_string,
            suffix_to_remove='.png'
        )
        # === 生成 captions（每行对应一个变量） ===
        if row_captions is None:
            captions = [[f"{flow} mL/s" for flow in self.flow_rates] for _ in image_names_first]
        else:
            captions = [
                row_captions for _ in image_names_first
            ]
        if main_captions is None:
            # === 生成 main_captions（主标题） ===
            main_captions = []
            for img_name in image_names_first:
                var, direc, val = self.latex.image_name_to_list(img_name)
                val_mm = f"{val * 1000:.2f}"
                main_captions.append(f"{direc}={val_mm}mm {var}云图")
        # === 生成最终 LaTeX 代码 ===
        res = f"{self.vape_name}不同{txt}如下图所示：\n"
        res += self.latex.insert_figures_per_row(
            image_names=image_names_matrix,
            captions=captions,
            main_captions=main_captions
        )
        return res

    def structural_design_analysis(self):
        code = ''
        code += '\\section{气道结构分析}\n'

        code += '\\begin{center}\n'
        code += f'\t\\textbf{{产品型号：{self.vape_name}，版本号：{self.version_number}}}\n'
        code += '\\end{center}\n\n'

        image_name_geo_inlet = 'geometry_inlet'
        code += (
            f'图\\ref{{fig: {image_name_geo_inlet}}}为进气口视图，'
            f'其总面积为{self.params_str.inlet_area}$\\text{{mm}}^2$\n\n'
        )
        code += self.latex.insert_figure(
            name=image_name_geo_inlet,
            title='进气口结构图',
            tag=image_name_geo_inlet,
            image_height=3.5
        )

        image_name_geo_airway = 'geometry_airway'
        code += (
            f'图\\ref{{fig: {image_name_geo_airway}}}为内部流道结构图，'
            f'该结构雾化区长度为{self.params_str.ato_length}$\\text{{mm}}$，'
            f'雾化区内径为{self.params_str.ato_diameter}$\\text{{mm}}$，'
            f'雾化点至出气口流道长度为{self.params_str.ato_to_outlet_length}$\\text{{mm}}$\n\n'
        )
        code += self.latex.insert_figure(
            name=image_name_geo_airway,
            title='内部流道',
            tag=image_name_geo_airway,
            image_height=10
        )

        code += '气道结构参数如下表所示：\n\n'
        table_data_structure = [
            ['结构参数', '数值'],
            ['进气口总面积($\\text{mm}^2$)', f'{self.params_str.inlet_area}'],
            ['雾化区长度($\\text{mm}$)', f'{self.params_str.ato_length}'],
            ['雾化区内径($\\text{mm}$)', f'{self.params_str.ato_diameter}'],
            ['雾化点至出气口流道长度($\\text{mm}$)', f'{self.params_str.ato_to_outlet_length}']
        ]
        code += self.latex.insert_table(
            data=table_data_structure,
            caption='气道结构参数表',
            label='airway_structure',
            alignment='ll',
            position='H'
        )
        return code

    def compared_with_experiments(self, case_names, flow_rates):
        code = ''
        code += '\\section{气道仿真结果与实验对比}\n'
        code += f'采用测试仪对{self.vape_name}进行吸阻测试，如图所示。\n\n'
        image_name_exp = 'experiment'
        code += self.latex.insert_figure(
            name=image_name_exp,
            title='吸阻测试',
            tag=image_name_exp,
            image_height=5
        )
        code += '气道仿真结果与实验对比如下图所示。\n\n'
        csv_exp = CSVUtils(self.pm.path_data, 'experiment_data')
        df_exp = csv_exp.read()
        sim_x_list = [flow_rates]
        sim_y_list = []
        sim_y = []
        for case_name in case_names:
            path_results = os.path.join(self.pm.path_simulation, case_name)
            results = GetParams().simulation(path_results)
            sim_y.append(results.delta_p_flow)
        sim_y_list.append(sim_y)
        image_manager = Image(
            self.path_latex_images,
            'draw_resistance_exp_vs_sim',
            x=sim_x_list,
            y=sim_y_list,
            label_x='Flow Rate(mL/s)',
            label_y='Draw Resistance(Pa)',
            figure_size=(14, 9)
        )
        image_manager.plt_curve_with_points(
            colors=['red'],
            scatter_x=[df_exp.iloc[:, 0]],
            scatter_y=[df_exp.iloc[:, 1]],
            y_range=[min(sim_y_list[0])-100, max(sim_y_list[0])+200],
            legend_text=['Simulation', 'Experiment'],
            linewidth=4
        )
        image_name_vs = 'draw_resistance_exp_vs_sim'
        code += self.latex.insert_figure(
            name=image_name_vs,
            title='吸阻对比曲线',
            tag=image_name_vs,
                image_height=7
        )
        return code

    def rans_spf_single(self, case_folder_name):
        code = ''
        code += self.latex.preamble()
        code += self.latex.first_page()
        code += '\\section{气道结构分析}\n'

        image_name_geo_inlet = 'geometry_inlet'
        code += (
            f'图\\ref{{fig: {image_name_geo_inlet}}}为进气口视图，'
            f'其总面积为{self.params_str.inlet_area}$\\text{{mm}}^2$\n\n'
        )
        code += self.latex.insert_figure(
            name=image_name_geo_inlet,
            title='进气口结构图',
            tag=image_name_geo_inlet,
            image_height=3.5
        )

        image_name_geo_airway = 'geometry_airway'
        code += (
            f'图\\ref{{fig: {image_name_geo_airway}}}为内部流道结构图，'
            f'该结构雾化区长度为{self.params_str.ato_length}$\\text{{mm}}$，'
            f'雾化区内径为{self.params_str.ato_diameter}$\\text{{mm}}$，'
            f'雾化点至出气口流道长度为{self.params_str.ato_to_outlet_length}$\\text{{mm}}$\n\n'
        )
        code += self.latex.insert_figure(
            name=image_name_geo_airway,
            title='内部流道',
            tag=image_name_geo_airway,
            image_height=10
        )

        code += '气道结构参数如下表所示：\n\n'
        table_data_structure = [
            ['结构参数', '数值'],
            ['进气口总面积($\\text{mm}^2$)', f'{self.params_str.inlet_area}'],
            ['雾化区长度($\\text{mm}$)', f'{self.params_str.ato_length}'],
            ['雾化区内径($\\text{mm}$)', f'{self.params_str.ato_diameter}'],
            ['雾化点至出气口流道长度($\\text{mm}$)', f'{self.params_str.ato_to_outlet_length}'],
        ]
        code += self.latex.insert_table(
            data=table_data_structure,
            caption='气道结构参数表',
            label='airway_structure',
            alignment='ll',  # 第一列左对齐，其余居中
            position='H'
        )

        code += ('\\section{气道仿真结果分析}\n')

        code += f'\n气道内流场计算所采用的抽吸条件为：{self.flow_rates:.2f}mL/s\n\n'

        code += '\\subsection{气道内流程特性分析}\n'

        path_sim = os.path.join(self.pm.path_simulation, case_folder_name)
        params_sim = GetParams().simulation(path_sim)

        code += '气道内流场主要物理特性如下表所示：\n\n'
        table_data_cfd_main = [
            ['物理量', '数值'],
            ['抽吸体积流率(mL/s)', f'{self.flow_rates:.2f}'],
            ['底部进气口平均速度(m/s)', f'{params_sim.velocity_ave_inlet:.2f}'],
            ['出气口平均速度(m/s)', f'{params_sim.velocity_ave_outlet:.2f}'],
            ['出气口平均总压($\\text{Pa}$)', f'{params_sim.total_pressure_ave_outlet:.2f}'],
            ['咪头位置平均总压($\\text{Pa}$)', f'{params_sim.total_pressure_ave_sensor:.2f}'],
            ['进出气口总压降($\\text{Pa}$)', f'{params_sim.delta_p_flow:.2f}'],
            ['咪头与出气口总压降($\\text{Pa}$)', f'{params_sim.delta_p_sensor:.2f}'],
        ]
        code += self.latex.insert_table(
            data=table_data_cfd_main,
            caption='气道内流场主要参数表',
            label='airway_flow',
            alignment='ll',  # 第一列左对齐，其余居中
            position='H'
        )

        code += f'简要说明：\n\n'
        code += self.results_analysis.total_pressure_loss(
            params_sim.delta_p_flow
        )
        code += self.results_analysis.sensor_pressure_loss(
            params_sim.delta_p_flow,
            params_sim.delta_p_sensor
        )

        code += '气道内流场其他特性如下表所示：\n\n'
        table_data_cfd_params = [
            ['物理量', '数值'],
            ['雾化区入口平均速度(m/s)', f'{params_sim.velocity_ave_ato_inlet:.2f}'],
            ['雾化区出口平均速度(m/s)', f'{params_sim.velocity_ave_ato_outlet:.2f}'],
            ['底部进气口平均总压($\\text{Pa}$)', f'{params_sim.total_pressure_ave_inlet:.2f}'],
            ['出气口平均总压($\\text{Pa}$)', f'{params_sim.total_pressure_ave_outlet:.2f}'],
            ['雾化区入口平均总压($\\text{Pa}$)', f'{params_sim.total_pressure_ave_ato_inlet:.2f}'],
            ['雾化区出口平均总压($\\text{Pa}$)', f'{params_sim.total_pressure_ave_ato_outlet:.2f}'],
            ['底部进气口平均静压($\\text{Pa}$)', f'{params_sim.pressure_ave_inlet:.2f}'],
            ['出气口平均静压($\\text{Pa}$)', f'{params_sim.pressure_ave_outlet:.2f}'],
            ['咪头位置平均静压($\\text{Pa}$)', f'{params_sim.pressure_ave_sensor:.2f}'],
            ['雾化区入口平均静压($\\text{Pa}$)', f'{params_sim.pressure_ave_ato_inlet:.2f}'],
            ['雾化区出口平均静压($\\text{Pa}$)', f'{params_sim.pressure_ave_ato_outlet:.2f}'],
            ['雾化区出入口总压降($\\text{Pa}$)', f'{params_sim.delta_p_ato:.2f}'],
            ['出气口平均湍动能($\\text{m}^2/\\text{s}^2$)', f'{params_sim.tke_ave_outlet:.4f}'],
            ['雾化区入口平均湍动能($\\text{m}^2/\\text{s}^2$)', f'{params_sim.tke_ave_ato_inlet:.4f}'],
            ['雾化区出口平均湍动能($\\text{m}^2/\\text{s}^2$)', f'{params_sim.tke_ave_ato_outlet:.4f}'],
            ['气道平均湍流强度', f'{params_sim.ti_ave_all:.4f}'],
            ['出气口平均湍流强度', f'{params_sim.ti_ave_outlet:.2f}'],
            ['雾化区入口平均湍流强度', f'{params_sim.ti_ave_ato_inlet:.4f}'],
            ['雾化区出口平均湍流强度', f'{params_sim.ti_ave_ato_outlet:.4f}'],
            ['Curle壁面平均声功率(dB)', f'{params_sim.curle_acoustic_power_ave:.2f}'],
            ['Curle壁面最大声功率(dB)', f'{params_sim.curle_acoustic_power_max:.2f}'],
            ['Proudman平均声功率(dB)', f'{params_sim.proudman_acoustic_power_ave:.2f}'],
            ['Proudman最大声功率(dB)', f'{params_sim.proudman_acoustic_power_max:.2f}']
        ]
        code += self.latex.insert_table(
            data=table_data_cfd_params,
            caption='气道内流场参数表',
            label='airway_flow',
            alignment='ll',
            position='H'
        )

        code += '\\subsection{总压分布}\n'
        code += self.insert_figures_single_sim('x向截面压力云图', 'total_pressure_x')
        code += self.insert_figures_single_sim('y向截面压力云图', 'total_pressure_y')
        code += self.insert_figures_single_sim('z向截面压力云图', 'total_pressure_z')
        code += '\\subsection{速度分布}\n'
        code += self.insert_figures_single_sim(
            'x向速度剖面',
            'velocity_multi_planex_view',
            main_captions=['x向速度剖面']
        )
        code += self.insert_figures_single_sim('x向截面速度云图', 'velocity_x')
        code += self.insert_figures_single_sim('y向截面速度云图', 'velocity_y')
        code += self.insert_figures_single_sim('z向截面速度云图', 'velocity_z')
        code += '\\subsection{湍动能分布}\n'
        code += self.insert_figures_single_sim('x向截面湍动能云图', 'turbulent_kinetic_energy_x')
        code += self.insert_figures_single_sim('y向截面湍动能云图', 'turbulent_kinetic_energy_y')
        code += self.insert_figures_single_sim('z向截面湍动能云图', 'turbulent_kinetic_energy_z')
        code += '\\subsection{涡量分布}\n'
        code += self.insert_figures_single_sim('x向截面涡量云图', 'vorticityx_x')
        code += self.insert_figures_single_sim('y向截面涡量云图', 'vorticityy_y')
        code += self.insert_figures_single_sim('z向截面涡量云图', 'vorticityz_z')
        code += '\\subsection{Q涡分布}\n'
        code += self.insert_figures_single_sim('x向截面Q涡量云图', 'qcriterion_x')
        code += self.insert_figures_single_sim('y向截面Q涡量云图', 'qcriterion_y')
        code += self.insert_figures_single_sim('z向截面Q涡量云图', 'qcriterion_z')
        code += '\\subsection{Proudman声功率分布}\n'
        code += self.insert_figures_single_sim('x向截面声功率云图', 'proudman_acoustic_power_x')
        code += self.insert_figures_single_sim('y向截面声功率云图', 'proudman_acoustic_power_y')
        code += self.insert_figures_single_sim('z向截面声功率云图', 'proudman_acoustic_power_z')

        code += '\\subsection{雾化区横截面速度分布}\n'
        code += self.insert_figures_single_sim('雾化区横截面速度分布', 'atomization_velocity')

        code += '\\subsection{流线}\n'
        images_stream_velocity = self.files_images.filter_star_filenames(
            'streamline_velocity',
            '.png'
        )
        images_stream_pressure = self.files_images.filter_star_filenames(
            'streamline_pressure',
            '.png'
        )
        images_stream = [
            item for pair in zip(
                images_stream_pressure,
                images_stream_velocity
            ) for item in pair
        ]
        print(images_stream)
        main_captions_stream = [
            'x向基于压力（左）及速度（右）流线图',
            'y向基于压力（左）及速度（右）流线图',
            'z向基于压力（左）及速度（右）流线图',
        ]
        code += self.latex.insert_figures_two_per_row(
            images_stream,
            captions=None,
            main_captions=main_captions_stream
        )

        code += '\\subsection{气道内表面压力分布}\n'
        images_pressure = self.files_images.filter_star_filenames(
            'contour_pressure',
            '.png'
        )
        images_wss = self.files_images.filter_star_filenames(
            'contour_wss',
            '.png'
        )
        images_contour = [item for pair in zip(images_pressure, images_wss) for item in pair]
        main_captions_pressure = [
            'x向表面压力云图（左）及壁面切应力云图（右）',
            'y向表面压力云图（左）及壁面切应力云图（右）',
            'z向表面压力云图（左）及壁面切应力云图（右）',
        ]
        code += self.latex.insert_figures_two_per_row(
            images_contour,
            captions=None,
            main_captions=main_captions_pressure
        )

        code += self.latex.end()
        self.latex.save_to_file(code)

        self.latex.compile_pdf(self.path_latex, PDFLATEXEXE)
        self.latex.compile_pdf(self.path_latex, PDFLATEXEXE)

    def rans_spf_comparison(self, case_names, column_names, captions=None):
        code = ''
        code += '\\section{气道内流场特性分析}\n'

        if isinstance(self.flow_rates, (int, float)):
            text_fr = f"{self.flow_rates}mL/s"
        else:
            text_fr = "、".join([f"{rate}mL/s" for rate in self.flow_rates])
        code += f'采用RANS方法对气道内流场进行计算，抽吸条件为{text_fr}。\n\n'

        # 初始化表头
        table_data_cfd_main = [
            [''] + [f'{c}' for c in column_names],
            ['底部进气口平均速度(m/s)'],
            ['出气口平均速度(m/s)'],
            ['出气口平均总压($\\text{{Pa}}$)'],
            ['咪头位置平均总压($\\text{{Pa}}$)'],
            ['进出气口总压降($\\text{{Pa}}$)'],
            ['咪头与出气口总压降($\\text{{Pa}}$)']
        ]

        # 初始化表头
        table_data_cfd_params = [
            [''] + [f'{c}' for c in column_names],
            ['雾化区入口平均速度(m/s)'],
            ['雾化区出口平均速度(m/s)'],
            ['底部进气口平均总压($\\text{Pa}$)'],
            ['出气口平均总压($\\text{Pa}$)'],
            ['雾化区入口平均总压($\\text{Pa}$)'],
            ['雾化区出口平均总压($\\text{Pa}$)'],
            ['底部进气口平均静压($\\text{Pa}$)'],
            ['出气口平均静压($\\text{Pa}$)'],
            ['咪头位置平均静压($\\text{Pa}$)'],
            ['雾化区入口平均静压($\\text{Pa}$)'],
            ['雾化区出口平均静压($\\text{Pa}$)'],
            ['雾化区出入口总压降($\\text{Pa}$)'],
            ['出气口平均湍动能($\\text{m}^2/\\text{s}^2$)'],
            ['雾化区入口平均湍动能($\\text{m}^2/\\text{s}^2$)'],
            ['雾化区出口平均湍动能($\\text{m}^2/\\text{s}^2$)'],
            ['气道平均湍流强度'],
            ['出气口平均湍流强度'],
            ['雾化区入口平均湍流强度'],
            ['雾化区出口平均湍流强度'],
            ['Curle壁面平均声功率(dB)'],
            ['Curle壁面最大声功率(dB)'],
            ['Proudman平均声功率(dB)'],
            ['Proudman最大声功率(dB)']
        ]

        total_pressure_sensors = []
        total_pressure_losses = []
        sensor_pressure_losses = []
        # 逐个流率计算并填入对应列
        for case_name in case_names:
            path_results = os.path.join(self.pm.path_simulation, case_name)
            results = GetParams().simulation(path_results)
            total_pressure_losses.append(results.delta_p_flow)
            sensor_pressure_losses.append(results.delta_p_sensor)
            total_pressure_sensors.append(results.total_pressure_ave_sensor)

            # 向每一行添加当前流率的计算结果
            table_data_cfd_main[1].append(f'{results.velocity_ave_inlet:.2f}')
            table_data_cfd_main[2].append(f'{results.velocity_ave_outlet:.2f}')
            table_data_cfd_main[3].append(f'{results.total_pressure_ave_outlet:.2f}')
            table_data_cfd_main[4].append(f'{results.total_pressure_ave_sensor:.2f}')
            table_data_cfd_main[5].append(f'{results.delta_p_flow:.2f}')
            table_data_cfd_main[6].append(f'{results.delta_p_sensor:.2f}')

            table_data_cfd_params[1].append(f'{results.velocity_ave_ato_inlet:.2f}')
            table_data_cfd_params[2].append(f'{results.velocity_ave_ato_outlet:.2f}')
            table_data_cfd_params[3].append(f'{results.total_pressure_ave_inlet:.2f}')
            table_data_cfd_params[4].append(f'{results.total_pressure_ave_outlet:.2f}')
            table_data_cfd_params[5].append(f'{results.total_pressure_ave_ato_inlet:.2f}')
            table_data_cfd_params[6].append(f'{results.total_pressure_ave_ato_outlet:.2f}')
            table_data_cfd_params[7].append(f'{results.pressure_ave_inlet:.2f}')
            table_data_cfd_params[8].append(f'{results.pressure_ave_outlet:.2f}')
            table_data_cfd_params[9].append(f'{results.pressure_ave_sensor:.2f}')
            table_data_cfd_params[10].append(f'{results.pressure_ave_ato_inlet:.2f}')
            table_data_cfd_params[11].append(f'{results.pressure_ave_ato_outlet:.2f}')
            table_data_cfd_params[12].append(f'{results.delta_p_ato:.2f}')
            table_data_cfd_params[13].append(f'{results.tke_ave_outlet:.4f}')
            table_data_cfd_params[14].append(f'{results.tke_ave_ato_inlet:.4f}')
            table_data_cfd_params[15].append(f'{results.tke_ave_ato_outlet:.4f}')
            table_data_cfd_params[16].append(f'{results.ti_ave_all:.4f}')
            table_data_cfd_params[17].append(f'{results.ti_ave_outlet:.4f}')
            table_data_cfd_params[18].append(f'{results.ti_ave_ato_inlet:.4f}')
            table_data_cfd_params[19].append(f'{results.ti_ave_ato_outlet:.4f}')
            table_data_cfd_params[20].append(f'{results.curle_acoustic_power_ave:.2f}')
            table_data_cfd_params[21].append(f'{results.curle_acoustic_power_max:.2f}')
            table_data_cfd_params[22].append(f'{results.proudman_acoustic_power_ave:.2f}')
            table_data_cfd_params[23].append(f'{results.proudman_acoustic_power_max:.2f}')

        code += '气道内流场主要物理特性如下表所示：\n\n'
        code += self.latex.insert_table(
            data=table_data_cfd_main,
            caption='气道内流场主要参数表',
            label='airway_flow',
            alignment='lccc',
            position='H'
        )

        code += '\\textbf{简要说明：}\n\n'
        code += self.results_analysis.default_flow_rates_pressure(
            total_pressure_sensors,
            total_pressure_losses,
            sensor_pressure_losses
        )

        code += '气道内流场其他特性如下表所示：\n\n'
        code += self.latex.insert_table(
            data=table_data_cfd_params,
            caption='气道内流场参数表',
            label='airway_flow',
            alignment='lccc',
            position='H'
        )

        code += '\\subsection{速度分布}\n'
        code += self.insert_figures_diff_sim('x向速度剖面', 'velocity_multi_planex_view', case_names, captions,
                                             main_captions=['x向速度剖面视图' for _ in range(2)])
        code += '\\subsubsection{x向速度分布}\n'
        code += self.insert_figures_diff_sim('x向截面速度云图', 'velocity_x', case_names, captions)
        code += '\\subsubsection{y向速度分布}\n'
        code += self.insert_figures_diff_sim('y向速度剖面', 'velocity_multi_planey_view', case_names, captions,
                                             main_captions=['y向速度剖面视图' for _ in range(2)])
        code += self.insert_figures_diff_sim('y向截面速度云图', 'velocity_y', case_names, captions)
        code += '\\subsubsection{z向速度分布}\n'
        code += self.insert_figures_diff_sim('z向速度剖面', 'velocity_multi_planez_view', case_names, captions,
                                             main_captions=['z向速度剖面视图' for _ in range(2)])
        code += self.insert_figures_diff_sim('z向截面速度云图', 'velocity_z', case_names, captions)

        code += '\\subsection{总压分布}\n'
        code += '\\subsubsection{x向总压分布}\n'
        code += self.insert_figures_diff_sim('x向截面压力云图', 'total_pressure_x', case_names, captions)
        code += '\\subsubsection{y向总压分布}\n'
        code += self.insert_figures_diff_sim('y向截面压力云图', 'total_pressure_y', case_names, captions)
        code += '\\subsubsection{z向总压分布}\n'
        code += self.insert_figures_diff_sim('z向截面压力云图', 'total_pressure_z', case_names, captions)

        code += '\\subsection{湍动能分布}\n'
        code += '\\subsubsection{x向湍动能分布}\n'
        code += self.insert_figures_diff_sim('x向截面湍动能云图', 'turbulent_kinetic_energy_x', case_names, captions)
        code += '\\subsubsection{y向湍动能分布}\n'
        code += self.insert_figures_diff_sim('y向截面湍动能云图', 'turbulent_kinetic_energy_y', case_names, captions)
        code += '\\subsubsection{z向湍动能分布}\n'
        code += self.insert_figures_diff_sim('z向截面湍动能云图', 'turbulent_kinetic_energy_z', case_names, captions)
        code += '\\subsection{涡量分布}\n'
        code += '\\subsubsection{x向涡量分布}\n'
        code += self.insert_figures_diff_sim('x向截面涡量云图', 'vorticityx_x', case_names, captions)
        code += '\\subsubsection{y向涡量分布}\n'
        code += self.insert_figures_diff_sim('y向截面涡量云图', 'vorticityy_y', case_names, captions)
        code += '\\subsubsection{z向涡量分布}\n'
        code += self.insert_figures_diff_sim('z向截面涡量云图', 'vorticityz_z', case_names, captions)
        code += '\\subsection{Q涡分布}\n'
        code += '\\subsubsection{x向Q涡分布}\n'
        code += self.insert_figures_diff_sim('x向截面Q涡量云图', 'qcriterion_x', case_names, captions)
        code += '\\subsubsection{y向Q涡分布}\n'
        code += self.insert_figures_diff_sim('y向截面Q涡量云图', 'qcriterion_y', case_names, captions)
        code += '\\subsubsection{z向Q涡分布}\n'
        code += self.insert_figures_diff_sim('z向截面Q涡量云图', 'qcriterion_z', case_names, captions)
        code += '\\subsection{Proudman声功率分布}\n'
        code += '\\subsubsection{x向声功率分布}\n'
        code += self.insert_figures_diff_sim('x向截面声功率云图', 'proudman_acoustic_power_x', case_names, captions)
        code += '\\subsubsection{y向声功率分布}\n'
        code += self.insert_figures_diff_sim('y向截面声功率云图', 'proudman_acoustic_power_y', case_names, captions)
        code += '\\subsubsection{z向声功率分布}\n'
        code += self.insert_figures_diff_sim('z向截面声功率云图', 'proudman_acoustic_power_z', case_names, captions)

        code += '\\subsection{雾化区横截面速度分布}\n'
        code += self.insert_figures_diff_sim('雾化区横截面速度分布', 'atomization_velocity', case_names, captions)

        code += '\\subsection{流线}\n'
        main_captions = [
            'x向基于速度流线图',
            'y向基于速度流线图',
            'z向基于速度流线图',
        ]
        code += self.insert_figures_diff_sim('基于速度流线图', 'streamline_velocity', case_names, row_captions=captions,
                                             main_captions=main_captions)
        code += '\\subsection{表面压力分布}\n'
        main_captions = [
            'x向表面压力云图',
            'y向表面压力云图',
            'z向表面压力云图',
        ]
        code += self.insert_figures_diff_sim('表面压力云图', 'contour_pressure', case_names, row_captions=captions,
                                             main_captions=main_captions)
        code += '\\subsection{壁面切应力分布}\n'
        main_captions = [
            'x向壁面切应力云图',
            'y向壁面切应力云图',
            'z向壁面切应力云图',
        ]
        code += self.insert_figures_diff_sim('壁面切应力云图', 'contour_wss', case_names, row_captions=captions,
                                             main_captions=main_captions)
        code += '\\subsection{Curle壁面声功率分布}\n'
        main_captions = [
            'x向Curle壁面声功率云图',
            'y向Curle壁面声功率云图',
            'z向Curle壁面声功率云图',
        ]
        code += self.insert_figures_diff_sim('Curle壁面声功率云图', 'contour_curle_acoustic_power', case_names,
                                             row_captions=captions, main_captions=main_captions)
        return code

    def rans_spf_comparison_flow_rates(
            self,
            user_name=None,
            bool_exp=False,
            all_flow_rates=None
    ):
        code = ''
        code += self.latex.preamble()
        code += self.latex.first_page()
        code += self.structural_design_analysis()
        if user_name is None:
            name = f'{self.version_number}'
        else:
            name = f'{self.version_number}_{user_name}'
        case_names = [
            f'{name}_rans_spf_q{fr}' for fr in self.flow_rates
        ]
        col_names = [
            f'{fr:.1f}(m/s)' for fr in self.flow_rates
        ]
        if bool_exp:
            all_case_names = [
                f'{name}_rans_spf_q{fr}' for fr in all_flow_rates
            ]
            code += self.compared_with_experiments(
                case_names=all_case_names,
                flow_rates=all_flow_rates
            )
        code += self.rans_spf_comparison(case_names, col_names)

        code += self.latex.end()
        self.latex.save_to_file(code)

        self.latex.compile_pdf(self.path_latex, PDFLATEXEXE)
        self.latex.compile_pdf(self.path_latex, PDFLATEXEXE)

    def rans_spf_comparison_user(
            self,
            case_names,
            col_names,
            captions
    ):
        code = ''
        code += self.latex.preamble()
        code += self.latex.first_page()
        code += self.structural_design_analysis()
        code += self.rans_spf_comparison(case_names, col_names, captions)

        code += self.latex.end()
        self.latex.save_to_file(code)

        self.latex.compile_pdf(self.path_latex, PDFLATEXEXE)
        self.latex.compile_pdf(self.path_latex, PDFLATEXEXE)

    def rans_spf_comparison_optimization(
            self,
            csv_id,
            all_case_names,
            compare_case_names,
            compare_column_names
    ):
        '''
        优化方案报告输出
        :return:
        '''
        code = ''
        code += self.latex.preamble()
        code += self.latex.first_page()
        code += '\\section{气道结构分析}\n'

        code += '\\begin{center}\n'
        code += f'\\textbf{{产品型号：{self.vape_name}，版本号：{self.version_number}}}\n\n'
        code += '\\end{center}\n'

        code += '\\section{气道内流场特性分析}\n'

        csv_input_params_name = f'{self.version_number}_input_params_{csv_id:02d}'
        csv_manager = CSVUtils(self.pm.path_data, csv_input_params_name)
        df_input_params = csv_manager.read()
        table_data_input_params = [
            [''] + [f'区域{i + 1}$\\Delta r$(mm)' for i in range(df_input_params.shape[1])]
        ]
        for i in range(len(df_input_params)):
            table_data_input_params.append(
                [f'{i+1}'] + [f'{val:.4f}' for val in df_input_params.iloc[i,:]]
            )
        code += '不同气道设计方案如下表所示：\n\n'
        code += self.latex.insert_table(
            data=table_data_input_params,
            caption='不同气道设计方案参数表',
            label='airway_flow',
            alignment='c'*(df_input_params.shape[1] + 1),
            position='H'
        )

        table_data_cfd_main = [
            [
                '',
                '$\\overline{U}_{\\text{in}}(\\mathrm{m/s})$',
                '$\\overline{p}_{\\text{out}}(\\mathrm{Pa})$',
                '$\\overline{p}_{\\text{sensor}}(\\mathrm{Pa})$',
                '$\\Delta p_{\\text{io}}(\\mathrm{Pa})$',
                '$\\Delta p_{\\text{so}}(\\mathrm{Pa})$',
                '$P_{\\text{s,curle,max}}(\\mathrm{dB})$',
                '$\\overline{T}_{\\text{u}}(\\mathrm{\\%})$'
            ]
        ]
        for i in range(len(all_case_names)):
            path_sim = os.path.join(self.pm.path_simulation, all_case_names[i])
            params_sim = GetParams().simulation(path_sim)
            table_data_cfd_main.append(
                [i+1] + [
                    f'{params_sim.velocity_ave_inlet:.2f}',
                    f'{params_sim.pressure_ave_outlet:.2f}',
                    f'{params_sim.pressure_ave_sensor:.2f}',
                    f'{params_sim.delta_p_flow:.2f}',
                    f'{params_sim.delta_p_sensor:.2f}',
                    f'{params_sim.curle_acoustic_power_max:.2f}',
                    f'{params_sim.ti_ave_all * 100:.2f}',
                ]
            )

        code += '不同气道设计方案内流场主要物理特性如下表所示：\n\n'
        code += self.latex.insert_table(
            data=table_data_cfd_main,
            caption='不同气道设计方案内流场主要参数表',
            label='airway_flow',
            alignment='cc',
            position='H'
        )

        code += (
            '其中，$\\overline{U}_{\\text{in}}$为底部进气口平均速度，'
            '$\\overline{p}_{\\text{out}}$为出气口平均总压，'
            '$\\overline{p}_{\\text{sensor}}$为咪头位置平均总压，'
            '$\\Delta p_{\\text{io}}$为进出气口总压降，'
            '$\\Delta p_{\\text{so}}$为咪头与出气口总压降，'
            '$P_{\\text{s,curle,max}}$为Curle壁面最大声功率，'
            '$\\overline{T}_{\\text{u}}$为气道平均湍流强度。'
        )

        # code += self.rans_spf_comparison(compare_case_names, compare_column_names)

        code += self.latex.end()
        self.latex.save_to_file(code)

        self.latex.compile_pdf(self.path_latex, PDFLATEXEXE)
        self.latex.compile_pdf(self.path_latex, PDFLATEXEXE)

if __name__ == '__main__':
    vape_name = 'ATN021'
    params_structure_dist = {
        'vape_type': '一次性',
        'inlet_area': 10,
        'outlet_area': 10,
        'ato_length': 11,
        'ato_diameter': 12,
        'ato_to_outlet_length': 13,
        'ato_inlet_coord': 1,
        'ato_outlet_coord': 1,
        'ato_direction': 'y'
    }
    flow_rate = 17.5
    path_root = r'D:\1_Work\active\202510_ATN021'
    folder_name = 'rans_flow_q17.5'
    path_simulation = os.path.join(path_root, 'simulation', folder_name)
    params_simulation = GetParams().simulation(path_simulation)
    params_structure = GetParams().structure(params_structure_dist)
    report = ReportLatex(path_root, folder_name, vape_name, params_structure, flow_rates=17.5)
    # report.convert_images()
    report.rans_spf_single(path_simulation)
    report_diff_flow_rate = ReportLatex(path_root, 'rans_flow_diff_flow_rate_comparison', vape_name, params_structure,
                                        flow_rates=[17.5, 20.5, 22.5])
    # report_diff_flow_rate.rans_flow_compare_latex()