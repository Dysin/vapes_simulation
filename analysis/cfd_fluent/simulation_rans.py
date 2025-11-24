'''
Fluent计算
Author: Dysin
Time:   2024.06.22
'''

from .configuration import Fluent
import numpy as np

def rans_flow(
    file_name,              # 文件名，网格与计算文件同名
    path_fluent,            # 计算路径
    path_mesh,              # 网格文件路径
    core_num,               # 并行核数
    bool_symmetry,          # 是否采用对称
    aoa,                    # 迎角
    beta,                   # 侧滑角
    density,                # 空气密度 [kg/m^3]
    viscosity,              # 空气粘度 [kg/(m·s)]
    turb_intensity,         # 湍流强度
    turb_length_scale,      # 湍流长度尺度
    turb_viscosity_ratio,   # 湍流粘度比
    inlet_names,            # 速度入口边界名
    velocity,               # 速度
    outlet_names,           # 压力出口边界名
    outlet_direction,       # 压力出口方向
    area_ref,               # 参考面积
    aircraft_wall_names,    # 飞行器边界名
    lift_direction,         # 升力方向
    drag_direction,         # 阻力方向
    mom_direction,          # 力矩方向
    mom_center,             # 力矩中心
    pressure,               # 操作压力
    iteration_step,         # 迭代步数
    ui_mode                 # 是否启动Fluent界面，默认为否
):
    fluent = Fluent(path_fluent, file_name, core_num, aoa, beta)
    fluent.launch(path_mesh, ui_mode)
    fluent.model()
    fluent.material_air(density, viscosity)
    fluent.bc_velocity_inlet(inlet_names, velocity, turb_intensity, turb_length_scale)
    fluent.bc_pressure_outlet(outlet_names, outlet_direction, turb_viscosity_ratio)
    if bool_symmetry:
        fluent.bc_symmetry('symmetry')
    fluent.reference_values(area_ref, density, viscosity, np.linalg.norm(velocity))
    fluent.report_forces(aircraft_wall_names, lift_direction, drag_direction, mom_direction, mom_center)
    fluent.residual()
    fluent.operation_conditions(pressure)
    fluent.initialize()
    fluent.calculate(iteration_step)