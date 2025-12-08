'''
@Desc:   Tecplot工具类
@Author: Dysin
@Time:   2024/8/6
'''

import os
import subprocess

class Tecplot_Base:
    def base_opt(self):
        file1 = (
            '#!MC 1410\n'
            '$!Interface ZoneBoundingBoxMode = Off\n'           # 隐藏模型虚线边框
            '$!ThreeDAxis FrameAxis{Show = No}\n'               # 隐藏坐标轴
            '$!FrameLayout ShowBorder = No\n'                   # 隐藏图形边框
            '$!GlobalContour 1  Legend{Box{BoxType = None}}\n'  # 隐藏图例边框
            # '$!GlobalContour 1  Legend{AutoResize = Yes}\n'     # 图例尺寸重构
            '$!GlobalContour 1  Legend{OverlayBarGrid = No}\n'  # 隐藏图例格子
            # '$!GlobalContour 1  Legend{NumberTextShape{FontFamily = \'Times New Roman\'}}\n'  # 设置图例字体
            '$!GlobalContour 1  Legend{Header{TextShape{FontFamily = \'Times New Roman\'}}}\n'  # 设置图例字体
            # '$!GlobalContour 1  Legend{NumberTextShape{SizeUnits = Point}}\n'
            '$!GlobalContour 1  Legend{NumberTextShape{Height = 1.4}}\n' # 图例尺寸
            '$!GlobalContour 1  Labels{NumFormat{Formatting = Exponential}}\n' # 图例数值采用科学计数
            '$!GlobalContour 1  Labels{NumFormat{Precision = 1}}\n' # 图例数值保留两位小数
        )
        file2 = self.close_all()
        return file1 + file2
    def close_all(self):
        file = (
            '$!FieldLayers ShowShade = No\n'
            '$!FieldLayers ShowContour = No\n'
            '$!FieldLayers ShowMesh = No\n'
            '$!FieldLayers UseTranslucency = No\n'
            '$!StreamtraceLayers Show = No\n'
            '$!IsoSurfaceLayers Show = No\n'
        )
        return file
    def run(self, path):
        path_mcr = os.path.join(path, 'post_all.mcr')
        command = f'tec360.exe -b -p {path_mcr}'
        print(f'[INFO] Run tecplot: {command}')
        subprocess.run(['cmd', '/c', command], check=True)

class Contour:
    def open(self):
        file = '$!FieldLayers ShowContour = Yes\n'  # 打开contour
        return file

    def close(self):
        file = '$!FieldLayers ShowContour = No\n'  # 关闭contour
        return file

    # 设置要显示的物理量序号var
    def set_var(self, var, group=1):
        file = (
            '$!SetContourVar\n'
            + '  Var = %s\n' % var
            + 'ContourGroup = %s\n' % group
            + 'LevelInitMode = ResetToNice\n'
        )
        return file

    # color: Small Rainbow, Sequential - Viridis
    def set_color(self, color='Small Rainbow', group=1):
        file = '$!GlobalContour %i  ColorMapName = \'%s\'\n' % (group, color)
        return file

    def color_reverse(self):
        file = '$!GlobalContour 1  ColorMapFilter{ReverseColorMap = Yes}\n'
        return file

    def continuous(self, bool_level_lines, field_map_num=None, group=1):
        if bool_level_lines:
            file = (
                '$!GlobalContour %i  ColorMapFilter{ColorMapDistribution = Continuous}\n' %group +
                '$!FieldMap [%i]  Contour{ContourType = BothLinesAndFlood}\n' %field_map_num +
                '$!GlobalContour 1  Labels{AutoLevelSkip = 1}\n'
            )
        else:
            file = (
                '$!GlobalContour %i  ColorMapFilter{ColorMapDistribution = Continuous}\n' %group +
                '$!GlobalContour 1  Labels{AutoLevelSkip = 1}\n'
            )
        return file

    def legend(self, **kwargs):
        group = kwargs.get('group', 1)
        show = kwargs.get('show', 'Yes')
        header = kwargs.get('header', 'Yes')
        file = (
            '$!GlobalContour %s  Legend{Show = %s}\n' % (group, show) +
            '$!GlobalContour %s  Legend{Header{Show = %s}}\n' % (group, header)
        )
        return file

    def show(self, zone_numbers, all_zone_number):
        file1 = ''
        file2 = ''
        for i in range(all_zone_number):
            file1 += '$!FieldMap [%i]  Contour{Show = No}\n' % (i + 1)
        for i in range(len(zone_numbers)):
            file2 += '$!FieldMap [%i]  Contour{Show = Yes}\n' % zone_numbers[i]  # 显示shade
        return file1 + file2

    # 设置图例取值范围
    def set_level(self, min_val, max_val, num, group=1):
        file2 = ''
        file1 = (
            '$!ContourLevels New\n'
            'ContourGroup = %i\n' % group +
            'RawData\n'
            '%i\n' % num
        )
        for i in range(num):
            file2 += '%.2f\n' % (min_val + i * (max_val - min_val) / (num - 1))
        file3 = (
            '$!GlobalContour %i  ColorMapFilter{ContinuousColor{CMin = %s}}\n' % (group, min_val)
            + '$!GlobalContour %i  ColorMapFilter{ContinuousColor{CMax = %s}}\n' % (group, max_val)
        )
        return file1 + file2 + file3

class Surface:
    '''
    表面
    '''
    def open(self):
        file = '$!FieldLayers ShowShade = Yes\n'  # 打开shade
        return file

    def close(self):
        file = '$!FieldLayers ShowShade = No\n'  # 关闭shade
        return file

    # 显示Shade
    def show(self, zone_numbers, all_zone_number):
        file1 = ''
        file2 = ''
        for i in range(all_zone_number):
            file1 += '$!FieldMap [%i]  Shade{Show = No}\n' % (i + 1)
        for i in range(len(zone_numbers)):
            file2 += '$!FieldMap [%i]  Shade{Show = Yes}\n' % zone_numbers[i]  # 显示shade
        return file1 + file2

    # 显示Surface
    def show_active(self, zone_numbers, all_zone_number):
        file = '$!ActiveFieldMaps -= [1-%s]\n' % all_zone_number
        for i in range(len(zone_numbers)):
            file += '$!ActiveFieldMaps += [%s]\n' % zone_numbers[i]  # 显示surface
        return file

class Slice:
    def __init__(self, position):
        '''
        截面
        :param position: 位置
        '''
        self.position = position

    def open(self):
        file = '$!SliceLayers Show = Yes\n'  # 打开slices
        return file

    def close(self):
        file = '$!SliceLayers Show = No\n'  # 关闭slices
        return file

    def plane_x(self, is_surface=None):
        file1 = (
            '$!SliceAttributes 1  EdgeLayer{Show = Yes}\n'
            '$!SliceAttributes 1  SliceSource = SurfaceZones\n'
        )
        file2 = (
            '$!SliceAttributes 1  SliceSurface = XPlanes\n' +
            '$!SliceAttributes 1  PrimaryPosition{X = %s}\n' % self.position +
            '$!ExtractSlices\n' +
            '  Group = 1\n' +
            '  TransientOperationMode = AllSolutionTimes\n'
        )
        if is_surface is None:
            return file2
        else:
            return file1 + file2

    def plane_y(self, is_surface=None):
        file1 = (
            '$!SliceAttributes 1  EdgeLayer{Show = Yes}\n'
            '$!SliceAttributes 1  SliceSource = SurfaceZones\n'
        )
        file2 = (
            '$!SliceAttributes 1  SliceSurface = YPlanes\n' +
            '$!SliceAttributes 1  PrimaryPosition{Y = %s}\n' % self.position +
            '$!ExtractSlices\n' +
            '  Group = 1\n' +
            '  TransientOperationMode = AllSolutionTimes\n'
        )
        if is_surface is None:
            return file2
        else:
            return file1 + file2

    def plane_z(self, is_surface=None):
        file1 = (
            '$!SliceAttributes 1  EdgeLayer{Show = Yes}\n'
            '$!SliceAttributes 1  SliceSource = SurfaceZones\n'
        )
        file2 = (
            '$!SliceAttributes 1  SliceSurface = ZPlanes\n' +
            '$!SliceAttributes 1  PrimaryPosition{Z = %s}\n' % self.position +
            '$!ExtractSlices\n' +
            '  Group = 1\n' +
            '  TransientOperationMode = AllSolutionTimes\n'
        )
        if is_surface is None:
            return file2
        else:
            return file1 + file2

# 视图方向
class View:
    def xy_plane(self):
        file = (
            '$!ThreeDView PSIAngle = 0\n'
            '$!ThreeDView ThetaAngle = 0\n'
            '$!ThreeDView AlphaAngle = 0\n'
        )
        return file

    def xz_plane(self):
        file = (
            '$!ThreeDView PSIAngle = 90\n'
            '$!ThreeDView ThetaAngle = 0\n'
            '$!ThreeDView AlphaAngle = 0\n'
        )
        return file

    def yz_plane(self):
        file = (
            '$!ThreeDView PSIAngle = 90\n'
            '$!ThreeDView ThetaAngle = -90\n'
            '$!ThreeDView AlphaAngle = 0\n'
        )
        return file

    def yz_inver_plane(self):
        file = (
            '$!ThreeDView PSIAngle = 90\n'
            '$!ThreeDView ThetaAngle = 90\n'
            '$!ThreeDView AlphaAngle = 0\n'
        )
        return file

    def user_def(self, position_x, position_y, position_z, view_width):
        file = (
            '$!ThreeDView\n'
            '  ViewerPosition\n'
            '    {\n'
            '    X = %s\n' % position_x +
            '    Y = %s\n' % position_y +
            '    Z = %s\n' % position_z +
            '    }\n'
            'ViewWidth = %s\n' % view_width
        )
        return file

    def user_def_all(self, angle_psi, angle_theta, angle_alpha, position_x, position_y, position_z, view_width):
        file = (
            '$!ThreeDView\n' +
            '  PSIAngle = %s' % angle_psi +
            '  ThetaAngle = %s' % angle_theta +
            '  AlphaAngle = %s' % angle_alpha +
            '  ViewerPosition\n'
            '    {\n'
            '    X = %s\n' % position_x +
            '    Y = %s\n' % position_y +
            '    Z = %s\n' % position_z +
            '    }\n'
            'ViewWidth = %s\n' % view_width
        )
        return file

    def user_def_angles(self, angle_psi, angle_theta, angle_alpha):
        file = (
            f'$!ThreeDView PSIAngle = {angle_psi}\n'
            f'$!ThreeDView ThetaAngle = {angle_theta}\n'
            f'$!ThreeDView AlphaAngle = {angle_alpha}\n'
        )
        return file

    def fit_everything(self):
        file = (
            '$!View Fit\n'
            '  ConsiderBlanking = Yes\n'
        )
        return file

    def open_lighting(self):
        file = '$!FieldLayers UseLightingEffect = Yes\n'
        return file

    def close_lighting(self):
        file = '$!FieldLayers UseLightingEffect = No\n'
        return file

    def default_vertical(self, width=10, height=8):
        file = (
            '$!GlobalContour 1  Legend{AnchorAlignment = MiddleCenter}\n'  # 设置图例方向
            '$!GlobalContour 1  Legend{IsVertical = No}\n'  # 设置图例为横向
            '$!GlobalContour 1  Legend{XYPos{X = 50}}\n'  # 设置图例位置
            '$!GlobalContour 1  Legend{XYPos{Y = 6}}\n'
            '$!FrameLayout XYPos{X = 0}\n'
            '$!FrameLayout XYPos{Y = 0}\n'
            '$!FrameLayout Width = %s\n' % width +
            '$!FrameLayout Height = %s\n' % height +
            '$!WorkspaceView FitAllFrames\n'
        )
        return file

    def default_horizontal(self, width=10, height=8):
        file = (
            '$!GlobalContour 1  Legend{AnchorAlignment = MiddleCenter}\n'  # 设置图例方向
            '$!GlobalContour 1  Legend{IsVertical = Yes}\n'  # 设置图例为横向
            '$!GlobalContour 1  Legend{XYPos{X = 94}}\n'  # 设置图例位置
            '$!GlobalContour 1  Legend{XYPos{Y = 50}}\n'
            '$!FrameLayout XYPos{X = 0}\n'
            '$!FrameLayout XYPos{Y = 0}\n'
            '$!FrameLayout Width = %s\n' % width +
            '$!FrameLayout Height = %s\n' % height +
            '$!WorkspaceView FitAllFrames\n'
        )
        return file

# 流线
class Streamtraces:
    def open(self, var_u):
        file = (
            '$!GlobalThreeDVector UVar = %s\n' % var_u[0] +
            '$!GlobalThreeDVector VVar = %s\n' % var_u[1] +
            '$!GlobalThreeDVector WVar = %s\n' % var_u[2] +
            '$!StreamtraceLayers Show = Yes\n'
        )
        return file

    def close(self):
        file = '$!StreamtraceLayers Show = No\n'
        return file

    def style_default(self):
        file = (
            '$!StreamAttributes AddArrows = No\n'
            '$!StreamAttributes LineThickness = 0.2\n'
            '$!StreamAttributes Color = Multi\n'
        )
        return file

    def create_volume_line(self, num_pts):
        file = (
            '$!Streamtrace Add\n' +
            '  DistributionRegion = SurfacesOfActiveZones\n' +
            '  NumPts = %i\n' % num_pts +
            '  StreamType = VolumeLine\n' +
            '  StreamDirection = Both\n'
        )
        return file

    def points_volume_line(self, num_pts, points_start_and_end, direction):
        file = (
            '$!Streamtrace Add\n'
            '  DistributionRegion = Rake\n'
            '  NumPts = %s\n' % num_pts +
            '  StreamType = VolumeLine\n'
            '  StreamDirection = %s\n' % direction +
            '  StartPos\n'
            '    {\n'
            '    X = %s\n' % points_start_and_end[0] +
            '    Y = %s\n' % points_start_and_end[2] +
            '    Z = %s\n' % points_start_and_end[4] +
            '    }\n'
            '  AltStartPos\n'
            '    {\n'
            '    X = %s\n' % points_start_and_end[1] +
            '    Y = %s\n' % points_start_and_end[3] +
            '    Z = %s\n' % points_start_and_end[5] +
            '    }\n'
        )
        return file

class Iso_Surface:
    def open(self):
        file = '$!IsoSurfaceLayers Show = Yes\n'
        return file

    def close(self):
        file = '$!IsoSurfaceLayers Show = No\n'
        return file

    def create(self, iso_value):
        if len(iso_value) == 1:
            file = (
                '$!IsoSurfaceAttributes 1  DefinitionContourGroup = 2\n' +
                '$!IsoSurfaceAttributes 1  Isovalue1 = %s\n' % iso_value[0] +
                '$!IsoSurfaceAttributes 1  Contour{FloodColoring = Group1}\n'
            )
        elif len(iso_value) == 2:
            file = (
                '$!IsoSurfaceAttributes 1  DefinitionContourGroup = 2\n' +
                '$!IsoSurfaceAttributes 1  IsoSurfaceSelection = TwoSpecificValues\n'
                '$!IsoSurfaceAttributes 1  Isovalue1 = %s\n' % iso_value[0] +
                '$!IsoSurfaceAttributes 1  Isovalue2 = %s\n' % iso_value[1] +
                '$!IsoSurfaceAttributes 1  Contour{FloodColoring = Group1}\n'
            )
        return file

    def translucency(self, is_translucency='No', value=None):
        if is_translucency == 'No':
            file = '$!IsoSurfaceAttributes 1  Effects{UseTranslucency = No}\n'
        else:
            file = (
                '$!IsoSurfaceAttributes 1  Effects{UseTranslucency = %s}\n' % is_translucency +
                '$!IsoSurfaceAttributes 1  Effects{SurfaceTranslucency = %s}\n' % value
            )
        return file


class Data:
    def line(self, path, file_name, point_star, point_end, num_pts):
        file = (
            '$!ExtendedCommand\n' +
            '  CommandProcessorID = \'Extract Precise Line\'\n' +
            '  Command = \'' +
            '  XSTART = %s' % point_star[0] +
            '  YSTART = %s' % point_star[1] +
            '  ZSTART = %s' % point_star[2] +
            '  XEND = %s' % point_end[0] +
            '  YEND = %s' % point_end[1] +
            '  ZEND = %s' % point_end[2] +
            '  NUMPTS = %i' % num_pts +
            f'  EXTRACTTHROUGHVOLUME = T EXTRACTTOFILE = T EXTRACTFILENAME = \\\'{path}\\{file_name}.dat\\\' \'\n'
        )
        return file

    def export(self, path, file_name, zone_number, var_number):
        file = (
            f'$!WriteDataSet  \"{path}\\{file_name}.dat\"\n'
            '  IncludeText = No\n' +
            '  IncludeGeom = No\n' +
            '  IncludeCustomLabels = No\n' +
            '  IncludeDataShareLinkage = No\n' +
            '  ZoneList =  [%s]\n' % zone_number +
            '  VarList =  [%s]\n' % var_number +
            '  Binary = No\n' +
            '  UsePointFormat = Yes\n' +
            '  Precision = 9\n' +
            '  TecplotVersionToWrite = TecplotCurrent\n'
        )
        return file


class Variables:
    def field(self, var_u, var_p):
        file = (
            '$!ExtendedCommand\n'
            '  CommandProcessorID = \'CFDAnalyzer4\'\n'
            '  Command = \'SetFieldVariables ConvectionVarsAreMomentum=\\\'F\\\' UVarNum=%s VVarNum=%s WVarNum=%s ID1=\\\'Pressure\\\' Variable1=%s ID2=\\\'NotUsed\\\' Variable2=0\'\n' % (var_u[0], var_u[1], var_u[2], var_p)
        )
        return file

    def calculate(self, function):
        file = (
            '$!ExtendedCommand\n'
            '  CommandProcessorID = \'CFDAnalyzer4\'\n'
            '  Command = \'Calculate Function=\\\'%s\\\' Normalization=\\\'None\\\' ValueLocation=\\\'Nodal\\\' CalculateOnDemand=\\\'T\\\' UseMorePointsForFEGradientCalculations=\\\'F\\\'\'\n' % function
        )
        return file


class Translucency:
    '''
    透明度
    '''
    def open(self):
        '''
        打开透明度显示
        :return:
        '''
        file = '$!FieldLayers UseTranslucency = Yes\n'
        return file
    def close(self):
        '''
        关闭透明度显示
        :return:
        '''
        file = '$!FieldLayers UseTranslucency = No\n'
        return file
    def show(self, zone_numbers, all_zone_number, translucency_value=50):
        '''
        透明度显示
        :param zone_numbers:        zone编号
        :param all_zone_number:     zone总数
        :param translucency_value:  透明度值
        :return:
        '''
        file1 = ''
        file2 = ''
        for i in range(all_zone_number):
            file1 += '$!FieldMap [%i]  Effects{UseTranslucency = No}\n' % (i + 1)
        for i in range(len(zone_numbers)):
            file2 += '$!FieldMap [%i]  Effects{UseTranslucency = Yes}\n' % zone_numbers[i]  # 显示shade
            file2 += '$!FieldMap [%i]  Effects{SurfaceTranslucency = %i}\n' % (zone_numbers[i], translucency_value)  # 显示shade
        return file1 + file2


class Lighting:
    def open(self):
        file = '$!FieldLayers UseLightingEffect = Yes\n'
        return file

    def close(self):
        file = '$!FieldLayers UseLightingEffect = No\n'
        return file

    def close_source(self):
        file = '$!GlobalThreeD LightSource{IncludeSpecular = No}\n'
        return file

class File:
    def read_fluent(self, path, file_name):
        file = (
            '$!ReadDataSet  \'"STANDARDSYNTAX" "1.0" "FILELIST_DATAFILES" "2" "%s/%s.cas.h5" "%s/%s.dat.h5"\'\n' % (path, file_name, path, file_name) +
            '  DataSetReader = \'Fluent Common Fluid FileUtils Loader\'\n'
            '  ReadDataOption = New\n'
            '  ResetStyle = Yes\n'
            '  AssignStrandIDs = Yes\n'
            '  InitialPlotType = Automatic\n'
            '  InitialPlotFirstZoneOnly = No\n'
            '  AddZonesToExistingStrands = No\n'
            '  VarLoadMode = ByName\n'
        )
        return file

    def read_tecplot(self, path, file_name, var_name_list):
        file = (
            '$!ReadDataSet  \'%s\\%s.plt\'\n' % (path, file_name) +
            '  ReadDataOption = New\n'
            '  ResetStyle = Yes\n'
            '  VarLoadMode = ByName\n'
            '  AssignStrandIDs = Yes\n'
            '  VarNameList = \'%s\'\n' % var_name_list
        )
        return file

    def add_stl(self, path, file_name):
        file = (
            '$!ReadDataSet  \'"StandardSyntax" "1.0" "FEALoaderVersion" "66051" "FILENAME_File" "%s\\%s" "Append" "Yes" "AutoAssignStrandIDs" "Yes"\'\n' % (path, file_name) +
            '  DataSetReader = \'3D Systems STL (FEA)\'\n'
        )
        return file

    def add_cas_and_dat(self, path, file_name, var_list):
        file = (
            '$!ReadDataSet  \'"STANDARDSYNTAX" "1.0" "FILELIST_DATAFILES" "2" "%s/%s.cas.h5" "%s/%s.dat.h5"\'\n' % (path, file_name, path, file_name) +
            '  DataSetReader = \'Fluent Common Fluid FileUtils Loader\'\n'
            '  VarNameList = \'%s\'\n' % var_list +
            '  ReadDataOption = Append\n'
            '  ResetStyle = No\n'
            '  AssignStrandIDs = Yes\n'
            '  InitialPlotType = Automatic\n'
            '  InitialPlotFirstZoneOnly = No\n'
            '  AddZonesToExistingStrands = No\n'
            '  VarLoadMode = ByName\n'
        )
        return file

    def list_in_dir(self, path, **kwargs):
        files = os.listdir(path)
        file_names = []
        if 'end_with_var' in kwargs:
            end_with_var = kwargs['end_with_var']
            for file in files:
                if file.endswith(end_with_var):
                    file_names.append(file.replace(end_with_var, ''))
            return file_names
        else:
            return files


class Save:
    def picture(self, path, file_name):
        file = (
            '$!ExportSetup ExportFormat = BMP\n' +
            '$!ExportSetup ImageWidth = 2000\n' +
            '$!ExportSetup ExportFName = \'%s\\%s.bmp\'\n' % (path, file_name) +
            '$!Export\n' +
            '  ExportRegion = AllFrames\n'
        )
        return file

    def video(self, path, file_name, time_scale):
        file = (
            '$!ExportSetup ExportFormat = MPEG4\n' +
            '$!ExportSetup ImageWidth = 2000\n' +
            '$!ExportSetup AnimationSpeed = 5\n' +
            '$!ExportSetup ExportFName = \'%s\\%s.mp4\'\n' % (path, file_name) +
            '$!AnimateTime\n'
            '  StartTime = %s\n' % time_scale[0] +
            '  EndTime = %s\n' % time_scale[1] +
            '  Skip = 1\n'
            '  CreateMovieFile = Yes\n'
            '  LimitScreenSpeed = Yes\n'
            '  MaxScreenSpeed = 10\n'
        )
        return file


class Time_Control:
    def all_zero(self, all_zone_number):
        file = (
            '$!ExtendedCommand\n'
            '  CommandProcessorID = \'Strand Editor\'\n'
            '  Command = \'ZoneSet=1-%s;AssignStrands=TRUE;StrandValue=0;AssignSolutionTime=FALSE;\'\n' % all_zone_number
        )
        return file



