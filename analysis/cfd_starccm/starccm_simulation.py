'''
@Desc:   
@Author: Dysin
@Date:   2025/10/9
'''

import os
import subprocess
from analysis.cfd_starccm.starccm_utils import StarCCMBuilder
from utils.global_variables import *
from utils.turbulence import *
from utils.units import UnitManager

class StarCCMSimulation:
    def __init__(self, path):
        self.path = path

    def run(self, np=8, new=True):
        macro_file = os.path.join(self.path, 'mcr.java')
        sim_file = os.path.join(self.path, 'init.sim')
        cmd = [str(STARCCMEXEC)]

        if new: # 无 sim 文件 => 新建仿真
            cmd.append("-new")
            cmd.extend([
                "-batch", str(macro_file),
                "-np", str(np)
            ])
            cmd.extend(["-locale", "en:US"])
        else:
            cmd.extend([
                "-batch", str(macro_file),
                "-np", str(np),
                str(sim_file)
            ])
        print("运行命令:", " ".join(cmd))
        subprocess.run(cmd, check=True)

    def rans_single_phase_flow(
            self,
            part_name,
            flow_rate,
            outlet_area,
            max_step=100,
            bool_gen_mesh=True,
            target_mesh_size = None,
            min_mesh_size = None,
            max_mesh_size = None,
            num_layers = None,
            atomization_area_pos = None,  # 雾化区出入口坐标，单位mm
            atomization_area_dir = 'y' # 雾化芯方向（入->出）
    ):
        unit_manager = UnitManager(density=1.18415)
        flow_rate_kg_s = unit_manager.convert(
            flow_rate,
            from_unit='mL/s',
            to_unit='kg/s'
        )
        velocity = unit_manager.convert(
            flow_rate,
            from_unit='mL/s',
            to_unit='m/s',
            area=outlet_area
        )
        print(f'[INFO] Velocity: {velocity}')
        turbulence = Turbulence_Params(velocity_ref=velocity, l_ref=3e-3)
        first_layer_y = turbulence.boundary_layer_first_y(y_plus=1)
        growth_rate = turbulence.boundary_layer_growth_rate(
            y_plus=1,
            last_layer_height=max_mesh_size,
            n_layers=num_layers
        )
        prism_thickness = turbulence.prism_thickness(1,growth_rate,num_layers)
        ccm_builder = StarCCMBuilder(self.path)
        txt = ''
        txt += ccm_builder.base()
        txt += ccm_builder.units('mm')
        txt += ccm_builder.units('m')

        if bool_gen_mesh:
            txt += ccm_builder.import_nas(self.path, part_name)
            txt += ccm_builder.get_mesh_part(part_name)
            txt += ccm_builder.get_part_surface(part_name, 'inlet')
            txt += ccm_builder.get_part_surface(part_name, 'outlet')
            # txt += ccm_builder.generate_mesh(
            #     part_name=part_name,
            #     target_size=target_mesh_size,
            #     min_size=min_mesh_size,
            #     num_layers=num_layers,
            #     growth_rate=growth_rate,
            #     prism_thickness=prism_thickness * 0.5,
            #     max_size=max_mesh_size
            # )
            txt += ccm_builder.automated_mesh(
                part_name=part_name,
                target_size=target_mesh_size,
                min_size=min_mesh_size,
                max_size=max_mesh_size,
                volume_growth_rate='FAST',
                bool_prism_layer=True,
                num_layers=num_layers,
                layer_total_thickness=prism_thickness * 0.5,
                first_layer_thickness=first_layer_y * 0.2,
                layer_disable_boundary_names=['inlet', 'outlet']
            )
        else:
            txt += ccm_builder.import_mesh(self.path, mesh_name=f'{part_name}.msh')
        txt += ccm_builder.physics_rans_flow()

        non_wall_names = [
            'inlet',
            'outlet'
        ]

        wall_names = [
            'airflow_sensor',
            'wall'
        ]

        boundary_names = non_wall_names + wall_names

        # value的第三个元素表示是否输出矢量的各分量
        function_names = {
            'pressure': ('Pressure', 'scale', False),
            'velocity': ('Velocity', 'vector', True),
            'tke': ('TurbulentKineticEnergy', 'scale', False),
            'wss': ('WallShearStress', 'vector', False),
            'qcriterion': ('Qcriterion', 'scale', False),
            'vorticity': ('VorticityVector', 'vector', True),
            'total_pressure': ('RelativeTotalPressure', 'scale', False),
            'curle_acoustic_power': ('CurleAcousticPowerDB', 'scale', False),
            'proudman_acoustic_power': ('ProudmanAcousticPowerDB', 'scale', False),
        }
        # user function 湍流强度
        txt += ccm_builder.user_field_function(
            'turbulence_intensity',
            'scale',
            'sqrt(max(2/3 * $TurbulentKineticEnergy, 0)) / max(mag($$Velocity), 1e-6)'
        )

        txt += ccm_builder.get_region(part_name)
        for boundary in boundary_names:
            txt += ccm_builder.get_boundary(part_name, boundary)

        txt += ccm_builder.boundary_stagnation_inlet('inlet', 0)
        txt += ccm_builder.boundary_mass_flow(
            boundary_name='outlet',
            mass_flow_rate=-flow_rate_kg_s
        )

        txt += ccm_builder.stopping_condition(max_steps=max_step)

        for key, values in function_names.items():
            txt += ccm_builder.get_field_function(
                key,
                values[0],
                values[1]
            )

        # 截面
        txt += ccm_builder.plane_section(
            section_name='atomization_area_inlet',
            direction=atomization_area_dir,
            position=atomization_area_pos[0] * 1.0e-3,  # 单位m
            region_name=part_name
        )
        txt += ccm_builder.plane_section(
            section_name='atomization_area_outlet',
            direction=atomization_area_dir,
            position=atomization_area_pos[1] * 1.0e-3,
            region_name=part_name
        )

        # 监控
        report_names = []
        for boundary in boundary_names:
            code, report_name = ccm_builder.report_surface_ave(
                'pressure',
                'scale',
                boundary
            )
            txt += code
            report_names.append(report_name)
            code, report_name = ccm_builder.report_surface_ave(
                'total_pressure',
                'scale',
                boundary
            )
            txt += code
            report_names.append(report_name)

        for boundary in non_wall_names:
            code, report_name = ccm_builder.report_surface_ave(
                'velocity',
                'vector',
                boundary
            )
            txt += code
            report_names.append(report_name)

        code, report_name = ccm_builder.report_surface_ave(
            'pressure',
            'scale',
            'atomization_area_inlet',
            'planeSection'
        )
        txt += code
        report_names.append(report_name)

        code, report_name = ccm_builder.report_surface_ave(
            'pressure',
            'scale',
            'atomization_area_outlet',
            'planeSection'
        )
        txt += code
        report_names.append(report_name)

        code, report_name = ccm_builder.report_surface_ave(
            'total_pressure',
            'scale',
            'atomization_area_inlet',
            'planeSection'
        )
        txt += code
        report_names.append(report_name)

        code, report_name = ccm_builder.report_surface_ave(
            'total_pressure',
            'scale',
            'atomization_area_outlet',
            'planeSection'
        )
        txt += code
        report_names.append(report_name)

        code, report_name = ccm_builder.report_surface_ave(
            'velocity',
            'vector',
            'atomization_area_inlet',
            'planeSection'
        )
        txt += code
        report_names.append(report_name)

        code, report_name = ccm_builder.report_surface_ave(
            'velocity',
            'vector',
            'atomization_area_outlet',
            'planeSection'
        )
        txt += code
        report_names.append(report_name)

        code, report_name = ccm_builder.report_volume_ave(
            'velocity',
            'vector',
            f'{part_name}'
        )
        txt += code
        report_names.append(report_name)

        code, report_name = ccm_builder.report_volume_ave(
            'pressure',
            'scale',
            f'{part_name}'
        )
        txt += code
        report_names.append(report_name)

        code, report_name = ccm_builder.report_volume_ave(
            'total_pressure',
            'scale',
            f'{part_name}'
        )
        txt += code
        report_names.append(report_name)

        code, report_name = ccm_builder.report_surface_ave(
            'turbulence_intensity',
            'user',
            'atomization_area_inlet',
            'planeSection'
        )
        txt += code
        report_names.append(report_name)

        code, report_name = ccm_builder.report_surface_ave(
            'turbulence_intensity',
            'user',
            'atomization_area_outlet',
            'planeSection'
        )
        txt += code
        report_names.append(report_name)

        code, report_name = ccm_builder.report_surface_ave(
            'turbulence_intensity',
            'user',
            'outlet'
        )
        txt += code
        report_names.append(report_name)

        code, report_name = ccm_builder.report_volume_ave(
            'turbulence_intensity',
            'user',
            f'{part_name}'
        )
        txt += code
        report_names.append(report_name)

        code, report_name = ccm_builder.report_surface_ave(
            'tke',
            'scale',
            'atomization_area_inlet',
            'planeSection'
        )
        txt += code
        report_names.append(report_name)

        code, report_name = ccm_builder.report_surface_ave(
            'tke',
            'scale',
            'atomization_area_outlet',
            'planeSection'
        )
        txt += code
        report_names.append(report_name)

        code, report_name = ccm_builder.report_surface_ave(
            'tke',
            'scale',
            'outlet'
        )
        txt += code
        report_names.append(report_name)

        # 气动噪声
        code, report_name = ccm_builder.report_surface_ave(
            'curle_acoustic_power',
            'scale',
            'wall'
        )
        txt += code
        report_names.append(report_name)
        code, report_name = ccm_builder.report_volume_ave(
            'proudman_acoustic_power',
            'scale',
            f'{part_name}'
        )
        txt += code
        report_names.append(report_name)

        boundary_vars = ['boundary_' + name for name in boundary_names]
        all_parts = [f'region_{part_name}']
        all_parts += boundary_vars
        for key, values in function_names.items():
            code, report_name = ccm_builder.report_max(
                key,
                values[1],
                all_parts,
                parts_type='mix',
                bool_component=values[2]
            )
            txt += code
            if values[2]:
                report_names += report_name
            else:
                report_names.append(report_name)
            code, report_name = ccm_builder.report_min(
                key,
                values[1],
                all_parts,
                parts_type='mix',
                bool_component=values[2]
            )
            txt += code
            if values[2]:
                report_names += report_name
            else:
                report_names.append(report_name)

        txt += ccm_builder.report_expression(
            'delta_p_flow',
            '${pressure_ave_inlet} - ${pressure_ave_outlet}'
        )
        report_names.append('delta_p_flow')
        txt += ccm_builder.report_expression(
            'delta_p_sensor',
            '${pressure_ave_airflow_sensor} - ${pressure_ave_outlet}'
        )
        report_names.append('delta_p_sensor')

        export_funcs = []
        for key, values in function_names.items():
            if values[1] == 'scale':
                export_funcs.append(f'primitiveFieldFunction_{key}')
            elif values[1] == 'vector':
                export_funcs.append(f'vectorMagnitudeFieldFunction_{key}')
                if values[2]:
                    export_funcs.append(f'vectorComponentFieldFunction_{key}0')
                    export_funcs.append(f'vectorComponentFieldFunction_{key}1')
                    export_funcs.append(f'vectorComponentFieldFunction_{key}2')
        print(f'[INFO] EXPORT NAME: {export_funcs}')

        txt += ccm_builder.initialize()
        txt += ccm_builder.run()
        for report_name in report_names:
            print(f'[INFO] {report_name}')
            txt += ccm_builder.report_export(report_name)
        txt += ccm_builder.save(part_name)
        txt += ccm_builder.export_tecplot(part_name, part_name, export_funcs)
        ccm_builder.write_mcrfile(content=txt)

    def electrothermal_coupling(
            self,
            part_name,
            temporality='steady',
            max_steps=1000,
            time_step=0.01,
            inner_itreations=5,
            max_time=None
    ):
        ccm_builder = StarCCMBuilder(self.path)
        txt = ''
        txt += ccm_builder.base()
        txt += ccm_builder.units('mm')
        txt += ccm_builder.units('m')

        txt += ccm_builder.import_nas(self.path, part_name)
        txt += ccm_builder.generate_solid_mesh(
            part_name=part_name,
            target_size=1.5,
            min_size=1,
            max_size=2
        )

        txt += ccm_builder.get_region(part_name)
        boundary_names = (
            'positive_electrode',
            'negative_electrode',
            'wall'
        )
        for boundary in boundary_names:
            txt += ccm_builder.get_boundary(part_name, boundary)

        txt += ccm_builder.physics_electrothermal_solid(temporality=temporality)
        txt += ccm_builder.material_solid(
            physics_name='solid',
            materail_name='al',
            density=1.2,
            electrical_conductivity=2.2,
            specific_heat=1,
            thermal_conductivity=5
        )

        txt += ccm_builder.boundary_wall_electrodynamics(
            boundary_name='positive_electrode',
            electrical_potential_condition='electric_potential',
            thermal_condition='heat_flux',
            electric_potential=2,
            electrical_resistance=1,
            thermal_value=0
        )

        txt += ccm_builder.boundary_wall_electrodynamics(
            boundary_name='negative_electrode',
            electrical_potential_condition='electric_potential',
            thermal_condition='heat_flux',
            electric_potential=0,
            electrical_resistance=1,
            thermal_value=0
        )

        txt += ccm_builder.boundary_wall_electrodynamics(
            boundary_name='wall',
            electrical_potential_condition='insulator',
            thermal_condition='heat_flux',
            thermal_value=0
        )

        txt += ccm_builder.stopping_condition(
            type=temporality,
            max_steps=max_steps,
            time_step=time_step,
            inner_iterations=inner_itreations,
            max_time=max_time
        )

        # value的第三个元素表示是否输出矢量的各分量
        function_names = {
            'temperature': ('Temperature', 'scale', False),
        }

        for key, values in function_names.items():
            txt += ccm_builder.get_field_function(
                key,
                values[0],
                values[1]
            )

        export_funcs = []
        for key, values in function_names.items():
            if values[1] == 'scale':
                export_funcs.append(f'primitiveFieldFunction_{key}')
            else:
                export_funcs.append(f'vectorMagnitudeFieldFunction_{key}')
                if values[2]:
                    export_funcs.append(f'vectorComponentFieldFunction_{key}0')
                    export_funcs.append(f'vectorComponentFieldFunction_{key}1')
                    export_funcs.append(f'vectorComponentFieldFunction_{key}2')
        print(f'[INFO] EXPORT NAME: {export_funcs}')

        txt += ccm_builder.initialize()
        txt += ccm_builder.run()
        txt += ccm_builder.save(part_name)
        txt += ccm_builder.export_tecplot(part_name, part_name, export_funcs)
        ccm_builder.write_mcrfile(content=txt)


if __name__ == '__main__':
    path_airway = r'D:\1_Work\templates\vapes_simulation\test_case\simulation\rans_spf_q17.5'
    path = r'D:\1_Work\active\heating_wire\test_case\simulation'
    starccm = StarCCMSimulation(path_airway)
    target_size = 3e-4
    min_size = 8e-5
    max_size = 2 * target_size
    num_layers = 10
    outlet_area = 33.07e-6  # 单位：m^2
    flow_rate = 17.5
    starccm.rans_single_phase_flow('VP353', flow_rate=flow_rate, outlet_area=outlet_area, max_step=500,
                                   bool_gen_mesh=True, target_mesh_size=target_size, min_mesh_size=min_size,
                                   max_mesh_size=max_size, num_layers=num_layers, atomization_area_pos=[-64, -53.2],
                                   atomization_area_dir='x')
    # starccm.run(new=True)
    # starccm.electrothermal_coupling(
    #     part_name='demo',
    #     temporality='steady',
    # )
    # starccm.run(new=True)