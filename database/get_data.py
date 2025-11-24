'''
@Desc:   获取结构设计参数、流场参数、口感参数等
@Author: Dysin
@Date:   2025/10/31
'''

import os
import csv
from utils.params_manager import *

def get_structure_data(path_project, model_number, flow_rates):
    csv_params_str = os.path.join(r'D:\1_Work\active\database', '结构参数表.csv')

    # 判断 model_number 是否已存在
    def generate_csv(file, header, row_data):
        file_exists = os.path.exists(file)
        model_exists = False
        if file_exists:
            with open(file, mode="r", encoding="utf-8-sig", newline="") as f:
                reader = csv.reader(f)
                next(reader, None)  # 跳过表头
                for row in reader:
                    if len(row) > 0 and row[0] == str(model_number):
                        model_exists = True
                        break
        # 如果不存在，则写入
        if not model_exists:
            with open(
                    file,
                    mode="w" if not file_exists else "a",
                    newline="",
                    encoding="utf-8-sig"
            ) as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(header)
                writer.writerow(row_data)

            action = "创建并写入" if not file_exists else "追加"
            print(f"操作成功：{action} 1 行数据到 {file}")
        else:
            print(f"跳过：型号 {model_number} 已存在于 {file}")

    # 结构参数表
    headers_str = [
        '产品型号',
        '进气口总面积(mm^2)',
        '出气口总面积(mm^2)',
        '雾化区长度(mm)',
        '雾化区内径(mm)',
        '雾化区出口至出气口流道长度(mm)'
    ]

    params_str = GetParams().structure_config(path_project)
    row_data_str = [
        model_number,
        params_str.inlet_area,
        f'{(params_str.outlet_area * 1.0e6):0.2f}',
        params_str.ato_length,
        params_str.ato_diameter,
        params_str.ato_to_outlet_length
    ]

    generate_csv(csv_params_str, headers_str, row_data_str)

    # 流场参数表
    header_sim = [
        '产品型号',
        '进气口平均速度(m/s)',
        '出气口平均速度(m/s)',
        '雾化区入口平均速度(m/s)',
        '雾化区出口平均速度(m/s)',
        '进气口平均静压(Pa)',
        '出气口平均静压(Pa)',
        '咪头位置平均静压(Pa)',
        '雾化区入口平均静压(Pa)',
        '雾化区出口平均静压(Pa)',
        '进气口平均总压(Pa)',
        '出气口平均总压(Pa)',
        '咪头位置平均总压(Pa)',
        '雾化区入口平均总压(Pa)',
        '雾化区出口平均总压(Pa)',
        '出气口平均湍动能(m^2/s^2)',
        '雾化区入口平均湍动能(m^2/s^2)',
        '雾化区出口平均湍动能(m^2/s^2)',
        '气道平均湍流强度',
        '出气口平均湍流强度',
        '雾化区入口平均湍流强度',
        '雾化区出口平均湍流强度',
        '进出气口总压降(Pa)',
        '咪头与出气口总压降(Pa)',
        '雾化区出入口总压降(Pa)',
        'Curle壁面平均声功率(dB)',
        'Curle壁面最大声功率(dB)',
        'Proudman平均声功率(dB)',
        'Proudman最大声功率(dB)'
    ]

    for flow_rate in flow_rates:
        csv_params_sim = os.path.join(r'D:\1_Work\active\database', f'气道流场参数表_Q{flow_rate}.csv')
        path_results = os.path.join(path_project, 'simulation', f'rans_flow_q{flow_rate}')
        params_sim = GetParams().simulation(path_results)
        row_data_sim = [
            model_number,
            params_sim.velocity_ave_inlet,
            params_sim.velocity_ave_outlet,
            params_sim.velocity_ave_ato_inlet,
            params_sim.velocity_ave_ato_outlet,
            params_sim.pressure_ave_inlet,
            params_sim.pressure_ave_outlet,
            params_sim.pressure_ave_sensor,
            params_sim.pressure_ave_ato_inlet,
            params_sim.pressure_ave_ato_outlet,
            params_sim.total_pressure_ave_inlet,
            params_sim.total_pressure_ave_outlet,
            params_sim.total_pressure_ave_sensor,
            params_sim.total_pressure_ave_ato_inlet,
            params_sim.total_pressure_ave_ato_outlet,
            params_sim.tke_ave_outlet,
            params_sim.tke_ave_ato_inlet,
            params_sim.tke_ave_ato_outlet,
            params_sim.ti_ave_all,
            params_sim.ti_ave_outlet,
            params_sim.ti_ave_ato_inlet,
            params_sim.ti_ave_ato_outlet,
            params_sim.delta_p_flow,
            params_sim.delta_p_sensor,
            params_sim.delta_p_ato,
            params_sim.curle_acoustic_power_ave,
            params_sim.curle_acoustic_power_max,
            params_sim.proudman_acoustic_power_ave,
            params_sim.proudman_acoustic_power_max,
        ]
        generate_csv(csv_params_sim, header_sim, row_data_sim)



if __name__ == '__main__':
    model_number = 'D581'
    proj_name = model_number.replace('-', '')
    print(proj_name)
    path_root = f'D:\\1_Work\\active\\project\\{proj_name}'
    get_structure_data(
        path_root,
        model_number,
        [17.5, 20.5, 22.5]
    )
    #