'''
@Desc:   Fluent结果采用Tecplot后处理
@Author: Dysin
@Time:   2024/8/6
'''

import numpy as np
import os.path
from utils import Files
from analysis.post_tecplot.tecplot_post import *
from analysis.cfd_starccm.starccm_data_analysis import StarCCMDataAnalysis
from geometry.geometry_utils import GeometryUtils

def unique_with_precision(value_list, min_prec=5, max_prec=7):
    """
    对 value_list 中的浮点数按指定精度进行 round，从 min_prec 开始逐步增加精度，
    直到结果不出现重复，或达到最大精度 max_prec。
    返回：处理后的列表（与输入长度一致）
    """
    # 输入检查（可选）
    if not value_list:
        return []
    for prec in range(min_prec, max_prec + 1):
        rounded_values = [round(v, prec) for v in value_list]
        # 如果无重复 → 返回当前精度结果
        if len(set(rounded_values)) == len(value_list):
            return rounded_values
    # 最终精度仍有重复 → 返回 max_prec 的结果
    return rounded_values

def starccm_post(
        path_geo, # 几何路径
        path_post, # 算例路径
        geo_name, # 几何名，后缀为stl
        file_name, # 文件名
        all_zone_number, # zone数
        show_zone_ids, # zone显示的id
        plane_views, # 平面角度
        atomization_area_pos, # 雾化区出入口坐标
        atomization_area_dir # 雾化区方向
):
    try:
        os.remove(os.path.join(path_post, 'post_all.mcr'))
    except:
        print('post_all.mcr is cleaned')

    def middle_range(scale, ratio_left=0.9, ratio_right=0.9):
        """
        在给定范围 scale = [min_x, max_x] 中取中间部分。
        默认去掉两端各 (1 - ratio)/2，比如 ratio=0.95 -> 各去掉5%。
        :param scale: [min_x, max_x]
        :return: [new_min, new_max]
        """
        min_x, max_x = scale
        total = max_x - min_x
        margin_left = (1 - ratio_left) / 2 * total
        margin_right = (1 - ratio_right) / 2 * total
        new_min = min_x + margin_left
        new_max = max_x - margin_right
        return [new_min, new_max]

    file = Files(path_post)
    file.create_folder(path_post)

    starccm_post = StarCCMDataAnalysis(path_post)
    pressure_min = starccm_post.get_value('pressure_min_parts')
    pressure_max = starccm_post.get_value('pressure_max_parts')
    pressure_ave = starccm_post.get_value('pressure_ave_parts')
    total_pressure_min = starccm_post.get_value('total_pressure_min_parts')
    total_pressure_ave = starccm_post.get_value('total_pressure_ave_parts')
    velocity_min = starccm_post.get_value('velocity_min_parts')
    velocity_max = starccm_post.get_value('velocity_max_parts')
    velocity_ave = starccm_post.get_value('velocity_ave_parts')
    vorticityx_min = starccm_post.get_value('vorticityx_min_parts')
    vorticityx_max = starccm_post.get_value('vorticityx_max_parts')
    vorticityy_min = starccm_post.get_value('vorticityy_min_parts')
    vorticityy_max = starccm_post.get_value('vorticityy_max_parts')
    vorticityz_min = starccm_post.get_value('vorticityz_min_parts')
    vorticityz_max = starccm_post.get_value('vorticityz_max_parts')

    # 获取几何包围盒尺寸
    geo_utils = GeometryUtils(path_geo)
    geo_params = geo_utils.get_stl_bounding_box(geo_name)
    geo_params_outlet = geo_utils.get_stl_bounding_box(f'{geo_name}_outlet')
    print(geo_params_outlet)

    var_list = '"X" "Y" "Z" "Curle Surface Acoustic Power dB" "Pressure" "Proudman Acoustic Power dB" "Q-Criterion" "Relative Total Pressure" "Turbulent Kinetic Energy" "Velocity: Magnitude" "Velocity[i]" "Velocity[j]" "Velocity[k]" "Vorticity: Magnitude" "Vorticity[i]" "Vorticity[j]" "Vorticity[k]" "Wall Shear Stress: Magnitude"'

    var_velocity = [11, 12, 13]

    # Contour：变量名,tecplot中的变量编号,最小值,最大值
    total_p_scale = middle_range([total_pressure_min, 0], ratio_left=0.8, ratio_right=0.8)
    non_wall_contours = [
        ['pressure', 5, pressure_ave - 30, pressure_ave + 30],
        ['proudman_acoustic_power', 6, 0, 60],
        ['qcriterion', 7, 0, 1e6],
        ['total_pressure', 8, total_pressure_ave - 30, total_pressure_ave + 30],
        ['turbulent_kinetic_energy', 9, 0, 5],
        ['velocity', 10, 0, 10],
        ['vorticity', 14, 0, 1e6],
        ['vorticityx', 15, -5000, 5000],
        ['vorticityy', 16, -5000, 5000],
        ['vorticityz', 17, -5000, 5000]
    ]

    wall_contours = [
        ['curle_acoustic_power', 4, 0, 80],
        ['pressure', 5, pressure_ave - 30, pressure_ave + 30],
        ['total_pressure', 8, total_pressure_ave - 30, total_pressure_ave + 30],
        ['turbulent_kinetic_energy', 9, 0, 5],
        ['wss', 18, 0, 2]
    ]

    content_all = ''
    content_all += read_plt(path_post, file_name, var_list)

    # 截面云图
    # X 方向
    x_scale = [geo_params['min_x'], geo_params['max_x']]
    x_scale = middle_range(x_scale)
    print(x_scale)
    x_plane_pos = list(np.linspace(
        x_scale[0],
        x_scale[1],
        8
    ))
    x_scale = [geo_params_outlet['min_x'], geo_params_outlet['max_x']]
    x_scale = middle_range(x_scale)
    print(x_scale)
    delta_x = x_scale[1] - x_scale[0]
    if delta_x > 1e-5:
        x_plane_pos += list(np.linspace(
            x_scale[0],
            x_scale[1],
            8
        ))
    x_plane_pos = unique_with_precision(x_plane_pos) # 默认保留五位小数
    content_all += slice_plane_xyz(
        path_save=path_post,
        all_zone_number=all_zone_number,
        contour_vars_and_levels=non_wall_contours,
        plane_positions=x_plane_pos,
        background_ratio=[8, 12],
        bool_level_lines=False,
        plane_direction='x',
        user_view_angles=plane_views[0],
        legend_positions='h'
    )
    all_zone_number += len(x_plane_pos)

    thd_views = [
        [40, -50, 10],
        [60, 240, 0]
    ]

    contourx_zone_ids = list(range(
        all_zone_number - len(x_plane_pos) + 1,
        all_zone_number + 1)
    )

    content_all += multi_plane_show(
        path_save=path_post,
        all_zone_number=all_zone_number,
        contour_vars_and_levels=non_wall_contours,
        background_ratio=[12, 12],
        bool_level_lines=False,
        user_view_angles=thd_views,
        contour_zone_numbers=contourx_zone_ids,
        show_zone_numbers=show_zone_ids,
        translucency_zone=show_zone_ids,
        translucency_value=80,
        plane_direction='x'
    )

    # Y 方向
    y_scale = [geo_params['min_y'], geo_params['max_y']]
    y_scale = middle_range(y_scale)
    y_plane_pos = list(np.linspace(
        y_scale[0],
        y_scale[1],
        8
    ))
    y_scale = [geo_params_outlet['min_y'], geo_params_outlet['max_y']]
    y_scale = middle_range(y_scale)
    delta_y = y_scale[1] - y_scale[0]
    if delta_y > 1e-5:
        y_scale = middle_range(y_scale)
        y_plane_pos += list(np.linspace(
            y_scale[0],
            y_scale[1],
            8
        ))
    y_plane_pos = unique_with_precision(y_plane_pos)
    content_all += slice_plane_xyz(
        path_post,
        all_zone_number,
        non_wall_contours,
        y_plane_pos,
        [8, 12],
        bool_level_lines=False,
        plane_direction='y',
        user_view_angles=plane_views[1],
        legend_positions='h'
    )
    all_zone_number += len(y_plane_pos)

    contoury_zone_ids = list(range(
        all_zone_number - len(y_plane_pos) + 1,
        all_zone_number + 1)
    )
    content_all += multi_plane_show(
        path_save=path_post,
        all_zone_number=all_zone_number,
        contour_vars_and_levels=non_wall_contours,
        background_ratio=[12, 12],
        bool_level_lines=False,
        user_view_angles=thd_views,
        contour_zone_numbers=contoury_zone_ids,
        show_zone_numbers=show_zone_ids,
        translucency_zone=show_zone_ids,
        translucency_value=80,
        plane_direction='y'
    )

    # Z 方向
    z_scale = [geo_params['min_z'], geo_params['max_z']]
    z_scale = middle_range(z_scale)
    z_plane_pos = list(np.linspace(
        z_scale[0],
        z_scale[1],
        8
    ))
    z_scale = [geo_params_outlet['min_z'], geo_params_outlet['max_z']]
    z_scale = middle_range(z_scale)
    delta_z = z_scale[1] - z_scale[0]
    if delta_z > 1e-5:
        z_scale = middle_range(z_scale)
        z_plane_pos += list(np.linspace(
            z_scale[0],
            z_scale[1],
            8
        ))
    z_plane_pos = unique_with_precision(z_plane_pos)
    content_all += slice_plane_xyz(
        path_post,
        all_zone_number,
        non_wall_contours,
        z_plane_pos,
        [8, 12],
        bool_level_lines=False,
        plane_direction='z',
        user_view_angles=plane_views[2],
        legend_positions='h'
    )
    all_zone_number += len(z_plane_pos)

    contourz_zone_ids = list(range(
        all_zone_number - len(z_plane_pos) + 1,
        all_zone_number + 1)
    )
    content_all += multi_plane_show(
        path_save=path_post,
        all_zone_number=all_zone_number,
        contour_vars_and_levels=non_wall_contours,
        background_ratio=[12, 12],
        bool_level_lines=False,
        user_view_angles=thd_views,
        contour_zone_numbers=contourz_zone_ids,
        show_zone_numbers=show_zone_ids,
        translucency_zone=show_zone_ids,
        translucency_value=80,
        plane_direction='z'
    )

    # 雾化区
    core_slices_number = 6
    core_scale = [
        atomization_area_pos[0] * 1.0e-3,
        atomization_area_pos[1] * 1.0e-3
    ]
    core_plane_pos = np.linspace(
        core_scale[0],
        core_scale[1],
        core_slices_number
    )
    core_plane_pos = np.round(core_plane_pos, 5)
    if atomization_area_dir == 'y' or atomization_area_dir == '-y':
        content_all += slice_plane_xyz(
            path_post,
            all_zone_number,
            non_wall_contours,
            core_plane_pos,
            [12, 12],
            bool_level_lines=True,
            plane_direction='y',
            user_view_angles=plane_views[1],
            legend_positions='h',
            file_name='atomization'
        )
    elif atomization_area_dir == 'z' or atomization_area_dir == '-z':
        content_all += slice_plane_xyz(
            path_post,
            all_zone_number,
            non_wall_contours,
            core_plane_pos,
            [12, 12],
            bool_level_lines=True,
            plane_direction='z',
            user_view_angles=plane_views[2],
            legend_positions='h',
            file_name='atomization'
        )
    else:
        content_all += slice_plane_xyz(
            path_post,
            all_zone_number,
            non_wall_contours,
            core_plane_pos,
            [12, 12],
            bool_level_lines=True,
            plane_direction='x',
            user_view_angles=plane_views[0],
            legend_positions='h',
            file_name='atomization'
        )
    all_zone_number += len(core_plane_pos)

    contourc_zone_ids = list(range(
        all_zone_number - len(core_plane_pos) + 1,
        all_zone_number + 1)
    )
    content_all += multi_plane_show(
        path_save=path_post,
        all_zone_number=all_zone_number,
        contour_vars_and_levels=non_wall_contours,
        background_ratio=[12, 12],
        bool_level_lines=False,
        user_view_angles=thd_views,
        contour_zone_numbers=contourc_zone_ids,
        show_zone_numbers=show_zone_ids,
        translucency_zone=show_zone_ids,
        translucency_value=80,
        plane_direction='c'
    )
    all_zone_number += len(core_plane_pos)

    # 表面压力云图
    user_views = [
        [90, 0, 0, 0.00399382, -0.297752, 0.00549178, 0.0853191]
    ]
    for wall_contour in wall_contours:
        content_all += surface_contour(
            path_post,      # 图片保存路径
            show_zone_ids,
            all_zone_number,
            wall_contour,
            user_views=None,
            legend_positions='h',
            plane_views=plane_views,
            background_ratio=[8,12]
        )

    # 流线图
    lines = [
        [0.045, 0.17, 0.264, 0.92, -0.01, -0.03],
        [0.045, 0.17, -0.264, -0.92, -0.01, -0.03],

    ]
    content_all += streamtrace(
        path_post,
        show_zone_ids,
        all_zone_number,
        non_wall_contours,
        plane_views=plane_views,
        user_views=None,
        var_velocity=var_velocity,
        points_start_and_end=None,
        translucency_value=80,
        streamlines_number=200,
        translucency_zone=show_zone_ids,
        legend_positions='h',
        bool_lighting=False,
        background_ratio=[8, 12]
    )
    write_mcrfile(path_post, content_all)

def vape_post(
        path_geo,
        path_post,
        geo_name,
        file_name,
        atomization_area_pos,
        atomization_area_dir='y'
):
    if atomization_area_dir == 'y':
        plane_views = [
            [90, -90, 90],  # x plane view
            [90, 180, -90],  # y plane view
            [0, 0, 0],  # z plane view
        ]
    elif atomization_area_dir == 'z':
        plane_views = [
            [90, -90, 0],  # x plane view
            [90, 0, 0],  # y plane view
            [0, 0, 0],  # z plane view
        ]
    elif atomization_area_dir == '-x':
        plane_views = [
            [90, 90, 180],  # x plane view
            [90, 180, 90],  # y plane view
            [0, 0, -90],  # z plane view
        ]
    else:
        plane_views = [
            [90, -90, 0],  # x plane view
            [90, 0, 0],  # y plane view
            [0, 0, 0],  # z plane view
        ]
    all_zone_number = 6
    show_zone_ids = [1, 2, 3, 4, 5, 6]
    starccm_post(
        path_geo=path_geo,
        path_post=path_post,
        geo_name=geo_name,
        file_name=file_name,
        all_zone_number=all_zone_number,
        show_zone_ids=show_zone_ids,
        plane_views=plane_views,
        atomization_area_pos=atomization_area_pos,
        atomization_area_dir=atomization_area_dir
    )
    Tecplot_Base().run(path_post)

if __name__ == '__main__':
    path_geo = r'D:\1_Work\active\202510_F06PP01\mesh'
    path = r'D:\1_Work\active\202510_F06PP01\simulation\rans_flow_small_inlet_q17.5'
    vape_post(
        path_geo=path_geo,
        path_post=path,
        geo_name='F06PP01_flow_small_inlet',
        file_name='F06PP01'
    )