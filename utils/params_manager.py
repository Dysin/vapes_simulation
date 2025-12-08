'''
@Desc:   参数管理
@Author: Dysin
@Date:   2025/10/24
'''

from analysis.cfd_starccm.starccm_data_analysis import StarCCMDataAnalysis
from dataclasses import dataclass
from utils.paths_manager import PathManager
from utils.files_utils import CSVUtils

# 计算结果参数
@dataclass
class SimulationParams:
    velocity_ave_inlet: float           # 进气口平均速度(m/s)
    velocity_ave_outlet: float          # 出气口平均速度(m/s)
    velocity_ave_ato_inlet: float       # 雾化区入口平均速度(m/s)
    velocity_ave_ato_outlet: float      # 雾化区出口平均速度(m/s)
    pressure_ave_inlet: float           # 进气口平均静压(Pa)
    pressure_ave_outlet: float          # 出气口平均静压(Pa)
    pressure_ave_sensor: float          # 咪头位置平均静压(Pa)
    pressure_ave_ato_inlet: float       # 雾化区入口平均静压(Pa)
    pressure_ave_ato_outlet: float      # 雾化区出口平均静压(Pa)
    total_pressure_ave_inlet: float     # 进气口平均总压(Pa)
    total_pressure_ave_outlet: float    # 出气口平均总压(Pa)
    total_pressure_ave_sensor: float    # 咪头位置平均总压(Pa)
    total_pressure_ave_ato_inlet: float # 雾化区入口平均总压(Pa)
    total_pressure_ave_ato_outlet: float# 雾化区出口平均总压(Pa)
    tke_ave_outlet: float               # 出气口平均湍动能(m^2/s^2)
    tke_ave_ato_inlet: float            # 雾化区入口平均湍动能(m^2/s^2)
    tke_ave_ato_outlet: float           # 雾化区出口平均湍动能(m^2/s^2)
    ti_ave_all: float                   # 气道平均湍流强度
    ti_ave_outlet: float                # 出气口平均湍流强度
    ti_ave_ato_inlet: float             # 雾化区入口平均湍流强度
    ti_ave_ato_outlet: float            # 雾化区出口平均湍流强度
    delta_p_flow: float                 # 进出气口总压降(Pa)
    delta_p_sensor: float               # 咪头与出气口总压降(Pa)
    delta_p_ato: float                  # 雾化区出入口总压降(Pa)
    curle_acoustic_power_ave: float     # Curle壁面平均声功率(dB)
    curle_acoustic_power_max: float     # Curle壁面最大声功率(dB)
    proudman_acoustic_power_ave: float  # Proudman平均声功率(dB)，湍流噪声
    proudman_acoustic_power_max: float  # Proudman最大声功率(dB)，湍流噪声

# 结构参数
@dataclass
class StructureParams:
    vape_type: str              # 电子烟类型
    inlet_area: float           # 进气口面积
    outlet_area: float          # 出气口面积
    ato_length: float           # 雾化区长度
    ato_diameter: float         # 雾化区内径
    ato_to_outlet_length: float # 雾化区出口至出气口长度
    ato_inlet_coord: float      # 雾化区入口坐标
    ato_outlet_coord: float     # 雾化区出口坐标
    ato_direction: float        # 雾化区轴心方向，x/y/z

@dataclass
class Names:
    project: str # 项目名
    vape: str # 产品型号
    geometry: str # 几何名
    mesh: str # 网格名
    simulation: str # 算例名

class GetParams:
    def simulation(self, path) -> SimulationParams:
        ccm_data = StarCCMDataAnalysis(path)
        # 提取原始数据
        results = {
            'velocity_ave_inlet': ccm_data.get_value('velocity_ave_inlet'),
            'velocity_ave_outlet': ccm_data.get_value('velocity_ave_outlet'),
            'velocity_ave_ato_inlet': ccm_data.get_value('velocity_ave_atomization_area_inlet'),
            'velocity_ave_ato_outlet': ccm_data.get_value('velocity_ave_atomization_area_outlet'),
            'pressure_ave_inlet': ccm_data.get_value('pressure_ave_inlet'),
            'pressure_ave_outlet': ccm_data.get_value('pressure_ave_outlet'),
            'pressure_ave_sensor': ccm_data.get_value('pressure_ave_airflow_sensor'),
            'pressure_ave_ato_inlet': ccm_data.get_value('pressure_ave_atomization_area_inlet'),
            'pressure_ave_ato_outlet': ccm_data.get_value('pressure_ave_atomization_area_outlet'),
            'total_pressure_ave_inlet': 0.0,  # 特殊处理值
            'total_pressure_ave_outlet': ccm_data.get_value('total_pressure_ave_outlet'),
            'total_pressure_ave_sensor': ccm_data.get_value('total_pressure_ave_airflow_sensor'),
            'total_pressure_ave_ato_inlet': ccm_data.get_value('total_pressure_ave_atomization_area_inlet'),
            'total_pressure_ave_ato_outlet': ccm_data.get_value('total_pressure_ave_atomization_area_outlet'),
            'tke_ave_outlet': ccm_data.get_value('tke_ave_outlet'),
            'tke_ave_ato_inlet': ccm_data.get_value('tke_ave_atomization_area_inlet'),
            'tke_ave_ato_outlet': ccm_data.get_value('tke_ave_atomization_area_outlet'),
            'ti_ave_all': ccm_data.get_value('turbulence_intensity_ave_parts'),
            'ti_ave_outlet': ccm_data.get_value('turbulence_intensity_ave_outlet'),
            'ti_ave_ato_inlet': ccm_data.get_value('turbulence_intensity_ave_atomization_area_inlet'),
            'ti_ave_ato_outlet': ccm_data.get_value('turbulence_intensity_ave_atomization_area_outlet'),
            'curle_acoustic_power_ave': ccm_data.get_value('curle_acoustic_power_ave_wall'),
            'curle_acoustic_power_max': ccm_data.get_value('curle_acoustic_power_max_parts'),
            'proudman_acoustic_power_ave': ccm_data.get_value('proudman_acoustic_power_ave_parts'),
            'proudman_acoustic_power_max': ccm_data.get_value('proudman_acoustic_power_max_parts'),
        }
        # 计算压力差值
        results['delta_p_flow'] = results['total_pressure_ave_inlet'] - results['total_pressure_ave_outlet']
        results['delta_p_sensor'] = results['total_pressure_ave_sensor'] - results['total_pressure_ave_outlet']
        results['delta_p_ato'] = results['total_pressure_ave_ato_inlet'] - results['total_pressure_ave_ato_outlet']
        return SimulationParams(**results)

    def names(self, results) -> Names:
        return Names(**results)

    def structure(self, results) -> StructureParams:
        return StructureParams(**results)

    def structure_config(self, path_root) -> StructureParams:
        '''
        从项目data中的configure获取结构参数
        :param path_root: 根目录
        :return:
        '''
        pm = PathManager(path_root)
        csv_manager = CSVUtils(pm.path_data, 'configure')
        df_config = csv_manager.read()
        inlet_area = float(df_config.iloc[0, 1])  # 进气口面积（mm^2）
        outlet_area = float(df_config.iloc[1, 1]) * 1.0e-6  # 出气口面积（m^2）
        atomizer_core_length = float(df_config.iloc[2, 1])  # 雾化区长度（mm）
        atomizer_core_diameter = float(df_config.iloc[3, 1])  # 雾化区内径（mm）
        core_to_outlet_length = float(df_config.iloc[4, 1])  # 雾化芯到出气口长度（mm）
        atomizer_core_pos_inlet = float(df_config.iloc[5, 1])
        atomizer_core_pos_outlet = float(df_config.iloc[6, 1])
        atomizer_core_dir = df_config.iloc[7, 1]
        vape_type = df_config.iloc[8, 1]
        results = {
            'vape_type': vape_type, # 电子烟类型
            'inlet_area': inlet_area,  # 进气口面积
            'outlet_area': outlet_area, # 出气口面积
            'ato_length': atomizer_core_length,  # 雾化区长度
            'ato_diameter': atomizer_core_diameter,  # 雾化区内径
            'ato_to_outlet_length': core_to_outlet_length,  # 雾化区出口至出气口长度
            'ato_inlet_coord': atomizer_core_pos_inlet,
            'ato_outlet_coord': atomizer_core_pos_outlet,
            'ato_direction': atomizer_core_dir,
        }
        return StructureParams(**results)