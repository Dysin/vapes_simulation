'''
@Desc:   post类
@Author: Dysin
@Time:   2024/8/6
'''

from analysis.cfd_tecplot.tecplot_utils import *

'''
全局变量
tecplot图例颜色:
    Sequential - Blue,
    Small Rainbow,
    Modified Rainbow - Dark ends,
'''
# LEGEND_COLOR = 'Modified Rainbow - Dark ends'
LEGEND_COLOR = 'Small Rainbow'

def slice_plane_xyz(
        path_save,                  # 图片保存路径
        all_zone_number,            # 总zone数
        contour_vars_and_levels,    # contour变量和范围
        plane_positions,            # 截面位置
        background_ratio,           # 背景图片比例
        bool_level_lines,           # 是否绘制level曲线
        plane_direction=None,       # 截面方向: 'x', 'y', 'z'
        user_view_angles=None,      # 用户自定义Angles
        file_name=None,
        views=None,
        show_zone_numbers=None,
        translucency_zone=None,
        translucency_value=50,
        time_scale=None,
        legend_positions='v',
        **kwargs
):
    '''
    x, y, z截面云图
    '''
    file_all = ''
    file_all += Tecplot_Base().base_opt()
    file_all += Contour().set_color(LEGEND_COLOR)
    file_all += Contour().open()
    level_num = 7
    if kwargs.get('colorReverse', 'No') == "Yes":
        file_all += Contour().color_reverse()
    file_all += Contour().legend(
        show=kwargs.get('legend_show', 'Yes'),
        header=kwargs.get('legend_header', 'No'))
    file_all += View().user_def_angles(
        user_view_angles[0],
        user_view_angles[1],
        user_view_angles[2]
    )
    file_all += View().close_lighting()
    if legend_positions == 'v':
        file_all += View().default_vertical(background_ratio[0], background_ratio[1])
    else:
        file_all += View().default_horizontal(background_ratio[0], background_ratio[1])
    for j, position in enumerate(plane_positions):
        if plane_direction == 'x':
            file_all += Slice(position).plane_x()
        elif plane_direction == 'y':
            file_all += Slice(position).plane_y()
        elif plane_direction == 'z':
            file_all += Slice(position).plane_z()
        all_zone_number += 1
        file_all += Contour().show([all_zone_number], all_zone_number)
        file_all += Contour().continuous(
            bool_level_lines=bool_level_lines,
            field_map_num=all_zone_number
        )
        if show_zone_numbers is not None:
            file_all += Surface().open()
            file_all += Surface().show(show_zone_numbers, all_zone_number)
        if translucency_zone is not None:
            file_all += Translucency().open()
            file_all += Translucency().show(translucency_zone, all_zone_number, translucency_value)
        for i, (var_name, var, level_min, level_max) in enumerate(contour_vars_and_levels):
            file_all += Contour().set_var(var)
            file_all += Contour().set_level(level_min, level_max, level_num)
            if views is None:
                file_all += View().fit_everything()
                if file_name is not None:
                    file_all += Save().picture(
                        path_save,
                        f"{file_name}_{var_name}_{plane_direction}{position}")
                else:
                    file_all += Save().picture(
                        path_save,
                        f"{var_name}_{plane_direction}{position}")
                if time_scale is not None:
                    file_all += Save().video(
                        path_save,
                        f"{var_name}_{plane_direction}{position}",
                        time_scale)
            else:
                for n_view, view in enumerate(views):
                    file_all += View().user_def(view[0], view[1], view[2], view[3])
                    if file_name is not None:
                        file_all += Save().picture(
                            path_save,
                            f"{var_name}_{plane_direction}{position}_view{n_view}_{file_name}")
                    else:
                        file_all += Save().picture(
                            path_save,
                            f"{var_name}_{plane_direction}{position}_view{n_view}")
                    if time_scale is not None:
                        file_all += Save().video(
                            path_save,
                            f"{var_name}_{plane_direction}{position}_view{n_view}",
                            time_scale)
    return file_all

def multi_plane_show(
        path_save,                  # 图片保存路径
        all_zone_number,            # 总zone数
        contour_vars_and_levels,    # contour变量和范围
        background_ratio,           # 背景图片比例
        bool_level_lines,           # 是否绘制level曲线
        user_view_angles=None,      # 用户自定义Angles
        file_name=None,
        contour_zone_numbers=None,
        show_zone_numbers=None,
        translucency_zone=None,
        translucency_value=50,
        time_scale=None,
        legend_positions='v',
        plane_direction='x',
        **kwargs
):
    '''
    x, y, z截面云图
    '''
    file_all = ''
    file_all += Tecplot_Base().base_opt()
    file_all += Contour().set_color(LEGEND_COLOR)
    file_all += Contour().open()
    level_num = 7
    if kwargs.get('colorReverse', 'No') == "Yes":
        file_all += Contour().color_reverse()
    file_all += Contour().legend(
        show=kwargs.get('legend_show', 'Yes'),
        header=kwargs.get('legend_header', 'No'))
    file_all += View().close_lighting()
    if legend_positions == 'v':
        file_all += View().default_vertical(background_ratio[0], background_ratio[1])
    else:
        file_all += View().default_horizontal(background_ratio[0], background_ratio[1])
    file_all += Contour().continuous(
        bool_level_lines=bool_level_lines,
        field_map_num=all_zone_number
    )
    file_all += Surface().open()
    file_all += Surface().show(show_zone_numbers, all_zone_number)
    file_all += Contour().show(contour_zone_numbers, all_zone_number)
    if translucency_zone is not None:
        file_all += Translucency().open()
        file_all += Translucency().show(translucency_zone, all_zone_number, translucency_value)
    for i, (var_name, var, level_min, level_max) in enumerate(contour_vars_and_levels):
        for j in range(len(user_view_angles)):
            file_all += Contour().set_var(var)
            file_all += Contour().set_level(level_min, level_max, level_num)
            file_all += View().user_def_angles(
                user_view_angles[j][0],
                user_view_angles[j][1],
                user_view_angles[j][2]
            )
            file_all += View().fit_everything()
            if file_name is not None:
                file_all += Save().picture(
                    path_save,
                    f"{file_name}_{var_name}_multi_plane{plane_direction}_view{j+1}")
            else:
                file_all += Save().picture(
                    path_save,
                    f"{var_name}_multi_plane{plane_direction}_view{j+1}")
            if time_scale is not None:
                file_all += Save().video(
                    path_save,
                    f"{var_name}_multi_plane{plane_direction}_view{j+1}",
                    time_scale)
    return file_all

def surface_contour(
        path_save,                  # 图片保存路径
        show_zone_numbers,          # 需显示zone编号
        all_zone_number,            # zone总数
        contour_vars_and_levels,    # contour变量和范围
        user_views=None,            # 视图位置
        legend_positions='v',       # legend位置
        plane_views=None,           # 自定义XYZ视图
        background_ratio=None,      # 图片比例
):
    '''
    表面云图
    '''
    file_all = ''
    file_all += Tecplot_Base().base_opt()
    file_all += View().open_lighting()
    if legend_positions == 'v':
        if background_ratio is None:
            file_all += View().default_vertical()
        else:
            file_all += View().default_vertical(background_ratio[0], background_ratio[1])
    else:
        if background_ratio is None:
            file_all += View().default_horizontal()
        else:
            file_all += View().default_horizontal(background_ratio[0], background_ratio[1])
    # Set contour properties
    file_all += Contour().set_color(LEGEND_COLOR)
    file_all += Contour().continuous(bool_level_lines=False)
    file_all += Contour().set_var(contour_vars_and_levels[1])
    file_all += Contour().set_level(contour_vars_and_levels[2], contour_vars_and_levels[3], 7)
    file_all += Contour().open()
    file_all += Contour().show(show_zone_numbers, all_zone_number)
    # Iterate through user views and save pictures
    if user_views is not None:
        for i, user_view in enumerate(user_views):
            file_all += View().user_def_all(
                user_view[0],
                user_view[1],
                user_view[2],
                user_view[3],
                user_view[4],
                user_view[5],
                user_view[6]
            )
            file_all += Save().picture(
                path_save,
                f'contour_{contour_vars_and_levels[0]}_{i}')
    else:
        file_all += View().user_def_angles(
            plane_views[0][0],
            plane_views[0][1],
            plane_views[0][2]
        )
        file_all += View().fit_everything()
        file_all += Save().picture(
            path_save,
            f'contour_{contour_vars_and_levels[0]}_{'z'}')
        file_all += View().user_def_angles(
            plane_views[1][0],
            plane_views[1][1],
            plane_views[1][2]
        )
        file_all += View().fit_everything()
        file_all += Save().picture(
            path_save,
            f'contour_{contour_vars_and_levels[0]}_{'y'}')
        file_all += View().user_def_angles(
            plane_views[2][0],
            plane_views[2][1],
            plane_views[2][2]
        )
        file_all += View().fit_everything()
        file_all += Save().picture(
            path_save,
            f'contour_{contour_vars_and_levels[0]}_{'x'}')
    return file_all

def streamtrace(
        path_save,                  # 图片保存路径
        show_zone_numbers,          # 需显示zone编号
        all_zone_number,            # zone总数
        contour_vars_and_levels,    # contour变量和范围
        var_velocity,               # 速度矢量u, v, w编号
        plane_views=None,           # XYZ视图角度
        user_views=None,            # 视图位置
        points_start_and_end=None,  # 线段两端点，[x1, x2, y1, y2, z1, z2]
        direction='Forward',        # 流线截取方向
        file_name=0,                # 文件名称区分号
        translucency_zone=None,     # 需透明zone
        translucency_value=50,      # 透明度值
        streamlines_number=100,     # 流线数
        legend_positions='v',       # 图例位置,
        bool_lighting=False,         # 是否开启lighting
        **kwargs
):
    '''
    流线图
    '''
    file_all = ''
    file_all += Tecplot_Base().base_opt()
    if 'legend_show' in kwargs and 'legend_header' in kwargs:
        file_all += Contour().legend(show=kwargs['legend_show'], header=kwargs['legend_header'])
    if 'background_ratio' in kwargs:
        background_ratio = kwargs['background_ratio']
        if legend_positions == 'v':
            file_all += View().default_vertical(background_ratio[0], background_ratio[1])
        else:
            file_all += View().default_horizontal(background_ratio[0], background_ratio[1])
    else:
        if legend_positions == 'v':
            file_all += View().default_vertical()
        else:
            file_all += View().default_horizontal()
    if bool_lighting:
        file_all += View().open_lighting()
    else:
        file_all += View().close_lighting()
    # file_all += View().fit_everything()
    file_all += Contour().set_color(LEGEND_COLOR)
    file_all += Contour().continuous(bool_level_lines=False)
    file_all += Surface().open()
    file_all += Surface().show(show_zone_numbers, all_zone_number)
    if points_start_and_end is None:
        file_all += Streamtraces().create_volume_line(100)
    else:
        for points in points_start_and_end:
            file_all += Streamtraces().points_volume_line(streamlines_number, points, direction)
    if translucency_zone is not None:
        file_all += Translucency().open()
        file_all += Translucency().show(translucency_zone, all_zone_number, translucency_value)
    file_all += Streamtraces().style_default()
    file_all += Streamtraces().open(var_velocity)
    if user_views is not None:
        for i, user_view in enumerate(user_views):
            file_all += View().user_def_all(
                user_view[0], user_view[1],
                user_view[2], user_view[3],
                user_view[4], user_view[5],
                user_view[6])
            for contour_var in contour_vars_and_levels:
                contour_name = contour_var[0]
                contour_vars = contour_var[1]
                contour_level_min = contour_var[2]
                contour_level_max = contour_var[3]
                file_all += Contour().set_var(contour_vars)
                file_all += Contour().set_level(contour_level_min, contour_level_max, 7)
                file_all += Save().picture(
                    path_save,
                    f'streamline_{contour_name}_view{i}_{file_name}')
    else:
        for i in range(len(plane_views)):
            file_all += View().user_def_angles(
                plane_views[i][0],
                plane_views[i][1],
                plane_views[i][2]
            )
            file_all += View().fit_everything()
            for contour_name, contour_vars, contour_level_min, contour_level_max in contour_vars_and_levels:
                file_all += Contour().set_var(contour_vars)
                file_all += Contour().set_level(contour_level_min, contour_level_max, 7)
                file_all += Save().picture(
                    path_save,
                    f"streamline_{contour_name}_view{i}_{file_name}"
                )
    return file_all

def iso_surface(
        path_save,                  # 图片保存路径
        show_zone_numbers,          # 需显示zone编号
        all_zone_number,            # zone总数
        contour_vars_and_levels,    # contour变量和范围
        user_views,                 # 视图位置
        variable,                   # iso surface的物理量
        var_numbers,                # var_numbers[size(2)], 0: 第一个变量值; 1: 第二个变量值
        iso_value,                  # iso变量值
        image_name='0',             # 图片编号区分名
        **kwargs
):
    '''
    绘制等值面
    **kwargs:
    :param: var_u:                  速度编号
    :param: var_p:                  压力编号
    :param: bool_iso_translucency:  等值面是否透明
    :param: iso_translucency_value: 等值面透明度值
    :param: translucency_zone:      透明区域
    :param: translucency_value:     透明度值
    :param: bool_legend_show:       是否显示图例
    :return:
    '''
    file_all = ''
    file_all += Tecplot_Base().base_opt()
    file_all += Lighting().open()
    file_all += Lighting().close_source()
    file_all += View().default_vertical()
    if 'var_u' in kwargs and 'var_p' in kwargs:
        var_u = kwargs['var_u']
        var_p = kwargs['var_p']
        file_all += Variables().field(var_u, var_p)
        file_all += Variables().calculate(variable)
    file_all += Contour().set_color(LEGEND_COLOR)
    file_all += Contour().continuous()
    file_all += Contour().set_var(var_numbers[0])
    file_all += Contour().set_level(contour_vars_and_levels[2], contour_vars_and_levels[3], 7)
    file_all += Contour().set_color(LEGEND_COLOR, 2)
    file_all += Contour().continuous(2)
    file_all += Contour().legend(group=2)
    file_all += Contour().set_var(var_numbers[1], 2)
    file_all += Surface().open()
    file_all += Surface().show(show_zone_numbers, all_zone_number)
    file_all += Iso_Surface().open()
    file_all += Iso_Surface().create(iso_value)
    if 'bool_legend_show' in kwargs:
        legend_show = kwargs['bool_legend_show']
        file_all += Contour().legend(show=legend_show)
    if 'bool_iso_translucency' in kwargs and 'iso_translucency_value' in kwargs:
        is_iso_translucency = kwargs['bool_iso_translucency']
        iso_translucency_value = kwargs['iso_translucency_value']
        file_all += Iso_Surface().translucency(is_translucency=is_iso_translucency, value=iso_translucency_value)
    if 'translucency_zone' in kwargs and 'translucency_value' in kwargs:
        translucency_zone = kwargs['translucency_zone']
        translucency_value = kwargs['translucency_value']
        file_all += Translucency().open()
        file_all += Translucency().show(translucency_zone, all_zone_number, translucency_value)
    for i, user_view in enumerate(user_views):
        file_all += View().user_def_all(user_view[0], user_view[1], user_view[2], user_view[3], user_view[4], user_view[5], user_view[6])
        file_all += Save().picture(path_save, f'iso_surface_{variable}_{i}_{image_name}')
        if 'time_scale' in kwargs:
            time_scale = kwargs['time_scale']
            file_all += Save().video(path_save, f'iso_surface_{variable}_{i}', time_scale)
    return file_all

def data_line(path_save, data_name, point_star, point_end, num_pts=300):
    '''
    线数据
    :param path_save:   数据保存路径
    :param data_name:   文件名
    :param point_star:  起始点
    :param point_end:   终止点
    :param num_pts:     一条线点数
    :return:
    '''
    file_all = Tecplot_Base().base_opt()
    for i in range(len(point_star)):
        file_all += Data().line(path_save, data_name[i], point_star[i], point_end[i], num_pts)
    return file_all

def data_surface(
        path_save,              # 图片保存路径
        show_zone_numbers,      # 需显示zone编号
        all_zone_number,        # zone总数
        plane_positions,        # 截面位置
        contour_vars,           # 变量
        file_name=None          # 保存文件名
):
    '''
    表面数据
    '''
    file_all = Tecplot_Base().base_opt()
    file_all += Surface().open()
    file_all += Surface().show_active(show_zone_numbers, all_zone_number)
    zone_number = all_zone_number
    for i in range(len(plane_positions)):
        zone_number += 1
        file_all += Slice(plane_positions[i]).plane_y('surface_zones')
        file_all += Data().export(path_save, f'data_{plane_positions[i]}_{file_name}', zone_number, contour_vars)
    return file_all

def read_cas_and_dat(path, file_name):
    '''
    读取Fluent文件
    :param path:        Fluent文件路径
    :param file_name:   文件名
    :return:
    '''
    file_all = '#!MC 1410\n'
    file_all += File().read_fluent(path, file_name)
    return file_all

def write_mcrfile(path_mcr, content):
    '''
    输出tecplot宏文件
    :param path_mcr:    文件路径
    :param content:     文件内容
    :return:
    '''
    file_mcr = os.path.join(path_mcr, 'post_all.mcr')
    with open(file_mcr, 'a') as tec_file:
        tec_file.write(content)

def calculate_var(var_velocity, var_pressure, variable):
    file_all = Variables().field(var_velocity, var_pressure)
    file_all += Variables().calculate(variable)
    return file_all

def add_stl_geom(path, file_name):
    file_all = File().add_stl(path, file_name)
    with open('post_add_stl.mcr', 'w') as tec_file:
        tec_file.write(file_all)

def add_fluent_dat(path, file_name, var_list):
    file_all = File().add_cas_and_dat(path, file_name, var_list)
    with open('post_add_fluent_dat.mcr', 'w') as tec_file:
        tec_file.write(file_all)

def time_all_zero(all_zone_number):
    file_all = Time_Control().all_zero(all_zone_number)
    with open('post_time_all_zero.mcr', 'w') as tec_file:
        tec_file.write(file_all)

def read_plt(path, file_name, var_list):
    file_all = '#!MC 1410\n'
    file_all += File().read_tecplot(path, file_name, var_list)
    return file_all
