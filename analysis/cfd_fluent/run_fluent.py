'''
@Desc:   运行Fluent
@Author: Dysin
@Time:   2024/8/11
'''

import os
import glob
from .simulation_rans import *
from utils import Direction_Transform
from utils import Design_Params_Data
from utils import Turbulence_Params

class Run_Fluent:
    def __init__(
            self,
            path_mesh,
            mesh_name,
            path_results,
            aircraft_wall_names,
            core_num,
            altitude
    ):
        '''
        运行Fluent
        :param path_mesh:               mesh文件路径
        :param path_results:            Fluent文件保存路径
        :param mesh_name:               网格名
        :param aircraft_wall_names:     飞行器表面名称
        :param core_num:                并行核数
        :param altitude:                飞行高度
        '''
        self.path_mesh = path_mesh
        self.path_results = path_results
        self.mesh_name = mesh_name
        self.aircraft_wall_names = aircraft_wall_names
        self.core_num = core_num
        self.altitude = altitude

    def rans(self, aoas, beta, iteration_step, bool_symmetry):
        n_row, data = Design_Params_Data().get_optimal_row_and_data()
        velocity_ref = data['巡航速度[m/s]']
        l_ref = data['平均气动弦长[m]']
        area_ref = data['机翼面积[m^2]']
        turb_params = Turbulence_Params(self.altitude, velocity_ref, l_ref)
        density = turb_params.rho
        viscosity = turb_params.mu()
        turb_intensity = turb_params.turbulence_intensity() * 100
        turb_length_scale = turb_params.turbulence_length_scale()
        turb_viscosity_ratio = turb_params.turbulence_viscosity_rate()
        pressure = turb_params.pressure()
        inlet_names = ['inlet']
        outlet_names = ['outlet']

        if bool_symmetry:
            area_ref = 0.5 * area_ref
        # velocity_ref = 30

        print('巡航速度[m/s]: ', velocity_ref)
        print('平均气动弦长[m]', l_ref)
        print('机翼面积[m^2]：', area_ref)

        for aoa in aoas:
            direction = Direction_Transform(aoa)
            velocity = direction.velocity(velocity_ref)
            # outlet_direction = direction.outlet()
            outlet_direction = [1, 0, 0]
            lift_direction = direction.lift()
            drag_direction = direction.drag()
            mom_direction = direction.moment_y()
            mom_center = [data['重心x坐标[m]'], 0, 0]

            files_trn = glob.glob(os.path.join('.', '*.trn'))
            files_bat = glob.glob(os.path.join('.', '*.bat'))
            files_log = glob.glob(os.path.join('.', '*.log'))
            files = files_bat + files_trn + files_log
            for file in files:
                try:
                    os.remove(file)
                except OSError as e:
                    print(f"Error deleting {file} : {e.strerror}")

            rans_flow(
                self.mesh_name,         # 文件名，网格与计算文件同名
                self.path_results,      # 计算路径
                self.path_mesh,         # 网格路径
                self.core_num,          # 并行核数
                bool_symmetry,          # 是否采用对称模型
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
                self.aircraft_wall_names,    # 飞行器边界名
                lift_direction,         # 升力方向
                drag_direction,         # 阻力方向
                mom_direction,          # 力矩方向
                mom_center,             # 力矩中心
                pressure,               # 操作压力
                iteration_step,         # 迭代步数
                # ui_mode='gui'           # 是否启动Fluent界面，默认为否
                ui_mode=None
            )