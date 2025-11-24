'''
@Desc:   ANSA功能实现
@Author: Dysin
@Date:   2024/8/4
'''

import os
import ansa
from ansa import *

class Geometry:
    def __init__(self, entities=None):
        '''
        几何类
        :param entities:    实体
        '''
        self.entities = entities
    def scale(self, scale_value):
        '''
        缩放
        :param scale_value: 缩放比例
        :return:
        '''
        print("Geometry scaling value: {0}".format(scale_value))
        ansa.base.GeoScale("MOVE", 0, "SAME PART", "EXPAND", 0, 0, 0, scale_value, self.entities, keep_connectivity=True)
    def delete_points_and_curves(self):
        print('[INFO BaseUtils] Delete hot points and curves...')
        ansa.base.DeleteVisibleHotPoints()
        ansa.base.DeleteCurves('all', True)
        ansa.base.PointsDelete('all')

    def create_line(self, point1, point2):
        x = [point1[0], point2[0]]
        y = [point1[1], point2[1]]
        z = [point1[2], point2[2]]
        curve = ansa.base.CreateCurve(2, x, y, z)
        print('[INFO] Create a straight line')
        return curve

class Entities:
    def __init__(self, type):
        '''
        实体类
        :param type:    "PSHELL", "FACE", "CONS", "POINTS", "CURVE", "HOT POINT"
        '''
        self.type = type
    def entities_visible(self):
        '''
        选择可视实体
        :return:
        '''
        return ansa.base.CollectEntities(ansa.constants.NASTRAN, None, self.type, filter_visible=True, prop_from_entities=False)
    def entities_all(self):
        '''
        选择全部实体
        :return:
        '''
        return ansa.base.CollectEntities(ansa.constants.NASTRAN, None, self.type, prop_from_entities=False)
    def pick_entity(self):
        '''
        选择单个实体
        :return:
        '''
        return ansa.base.PickEntities(ansa.constants.NASTRAN, self.type)
    def get_entity(self, id):
        '''
        根据ID获取ENTITY
        :param id: ID
        :return:
        '''
        entity = ansa.base.GetEntity(ansa.constants.NASTRAN, self.type, id)
        return entity
    def delete(self, entity):
        ansa.base.DeleteEntity(entity, True)

    def get_face_orientation(self, face_id):
        '''
        获取面法向
        :param face_id: 面id
        :return:
        '''
        face = self.get_entity(face_id)
        normal = ansa.base.GetFaceOrientation(face)
        return normal

class Measurement:
    def __init__(self, entity):
        '''
        测量类
        :param entity:  实体
        '''
        self.entity = entity
    def measurement_entity(self):
        '''
        测量可视实体
        :return:
        '''
        msr = ansa.base.CreateMeasurement(self.entity)
        ret_mes = ansa.base.GetEntityCardValues(ansa.constants.NASTRAN, msr, ('RESULT', 'RES1', 'RES2', 'RES3'))
        ansa.base.DeleteEntity(msr)
        return ret_mes
    def point_info(self):
        ret_mes = self.measurement_entity()
        result_mes = [ret_mes['RES1'], ret_mes['RES2'], ret_mes['RES3']]
        return result_mes
    def entity_info(self):
        ret_mes = self.measurement_entity()
        result_mes = ret_mes['RESULT']
        return result_mes

class Mesh:
    '''
    网格类
    '''
    def aspacing_stl(self, chordal_dev, min_value, max_value):
        ansa.mesh.AspacingSTL(chordal_dev, max_value, 0, min_value)

    def aspacing_cfd(self, min_value, max_value):
        ansa.mesh.AspacingCFD(1.2, 10.0, min_value, max_value, 15.0, 2.0)

    def reconstruct_shells(self, shells, min_value, max_value):
        self.aspacing_cfd(min_value, max_value)
        ansa.mesh.ReconstructShells(shells)

    def generate_boundary_layers(
            self,
            pshells,
            layers_id,
            first_height,
            growth_rate,
            layers_num
    ):
        '''
        生成边界层
        :param pshells:         需生成边界层的边界面
        :param layers_id:       生成边界层后的id
        :param first_height:    第一层网格高度
        :param growth_rate:     网格增长率
        :param layers_num:      边界层数
        :return:
        '''
        scenario = ansa.batchmesh.GetNewLayersScenario('layers_scenario')
        ret_scen = ansa.batchmesh.AddPartToMeshingScenario(pshells, scenario)
        session = ansa.base.NameToEnts('Default_Session')
        # 多边界层
        # layers_ses = []
        # session = ansa.batchmesh.GetNewLayersSession('layers_session')
        # ret_sess = ansa.batchmesh.AddPartToSession(pshells, session)
        # layers_ses.append(session)
        # ansa.batchmesh.AddSessionToMeshingScenario(layers_ses, scen)
        area = ansa.batchmesh.GetNewLayersArea(
            session[0],
            "area_params",
            first_height,
            "absolute",
            growth_rate,
            True,
            "layers_prop"
        )
        status = ansa.batchmesh.RunMeshingScenario(scenario)
        entity = Entities('PSOLID')
        layers_prop = entity.get_entity(layers_id)
        ansa.mesh.Redistribute(
            property=layers_prop,
            num_layers=layers_num,
            growth_rate=growth_rate
        )

    def create_mesh_refinement(self, min_coords, max_coords, max_surf_size, max_vol_size):
        '''
        创建网格加密区，根据最大最小坐标点创建
        :param min_coords:      最小坐标点
        :param max_coords:      最大坐标点
        :param max_surf_size:   最大面网格尺寸
        :param max_vol_size:    最大体网格尺寸
        :return:
        '''
        size_box = ansa.base.SizeBoxMinMax(
            None,
            min_coords,
            max_coords,
            max_surf_size,
            max_vol_size
        )
        return size_box

    def generate_volume_mesh(self, mesh_type):
        '''
        生成体网格
        :param mesh_type: "TETRA RAPID", "TETRA FEM", "TETRA CFD", "HEXA INTERIOR" or "HEXAPOLY"
        :return:
        '''
        print('[INFO BaseUtils] Detect volume!')
        vols = ansa.mesh.VolumesDetect(1, return_volumes=True)
        print('[INFO BaseUtils] Generate [%s] mesh!' %mesh_type)
        ansa.mesh.VolumesMeshV(vols[0], "%s" %mesh_type)

    def output(self, path, file_name, type):
        if type == 'Fluent' or type == 'cfd_fluent' or type == 'msh':
            ansa.base.OutputFluent(
                filename = os.path.join(path, file_name + '.msh'),
                mode = 'visible',
                format = 'ascii',
                scale = 0.001
            )
        elif type == 'nas':
            ansa.base.OutputNastran(
                filename = os.path.join(path, file_name + '.nas'),
                disregard_includes = 'on'
            )
        elif type == 'stl':
            ansa.base.OutputStereoLithography(
                filename=os.path.join(path, file_name + '.stl'),
                mode = 'visible',
                format = 'ascii'
            )

class MorphUtils:
    def morph_status(self, bool_active):
        '''
        设置Morphing状态，active/inactive
        :return:
        '''
        if bool_active:
            flag = ansa.morph.MorphFlagStatus('MORPHING_FLAG', True)
        else:
            flag = ansa.morph.MorphFlagStatus('MORPHING_FLAG', False)
        if flag == 0:
            print('[INFO] Morphing inactive!')
        else:
            print('[INFO] Morphing active!')

    def create_cylindrical_box(self, curve, radius1, radius2):
        morph_cyl = ansa.morph.MorphCylindrical(curve, radius1, radius2)
        res = ansa.morph.MorphLoad(morph_cyl, None, 'Visib')
        print(res)
        print('[INFO] Create cylinder box!')
        return morph_cyl

    def morph_cylindrical(self, morph_id, radius):
        entities_morph_faces = Entities('MORPHFACE')
        entities_param = Entities('PARAMETERS')
        moface_id = int((morph_id - 1) * 6)
        mofaces = []
        mofaces.append(entities_morph_faces.get_entity(moface_id + 1))
        mofaces.append(entities_morph_faces.get_entity(moface_id + 2))
        mofaces.append(entities_morph_faces.get_entity(moface_id + 3))
        mofaces.append(entities_morph_faces.get_entity(moface_id + 4))
        mofaces.append(entities_morph_faces.get_entity(moface_id + 5))
        mofaces.append(entities_morph_faces.get_entity(moface_id + 6))
        param_id = ansa.morph.MorphParamCreateRadiusOuter("Radius outer", mofaces)

        print(param_id)
        print('00000')
        param = entities_param.get_entity(param_id)
        print(param)
        res = ansa.morph.MorphParam(param, radius)
        print(res)
class BaseUtils:
    def checkss_templates(self, path):
        file_check = os.path.join(path, 'checks.plist')
        ansa.base.ReadCheckTemplatesFromFile(file_check)
        results = ansa.base.ExecuteCheckTemplate('stl_mesh', 0)
        print(results)

    def input_mesh(self, path, file_name, file_type):
        '''
        导入文件
        :param path:        文件路径
        :param file_name:   文件名
        :param file_type:   文件类型
        :return:
        '''
        print('[INFO BaseUtils] Input file %s/%s' % (path, file_name + '.' + file_type))
        file_mesh = os.path.join(path, file_name + '.' + file_type)
        if file_type == 'nas':
            ansa.base.InputNastran(file_mesh)

    def open_file(self, path, file_name):
        '''
        打开文件，几何或ANSA
        :param path:        文件路径
        :param file_name:   文件名
        :param file_type:   文件类型
        :return:
        '''
        print('[INFO BaseUtils] Open file %s/%s' % (path, file_name + '.ansa'))
        ansa.base.Open(os.path.join(path, file_name + '.ansa'))

    def save_file(self, path, file_name):
        '''
        保存ANSA文件，*.cfd_ansa
        :param path:        文件路径
        :param file_name:   文件名
        :return:
        '''
        ansa.base.SaveAs(os.path.join(path, file_name + '.cfd_ansa'))
    def new_pid(self, pid, name):
        '''
        新建PID
        :param pid:         pid
        :param name:        PID名称
        :return:            shell_property
        '''
        vals_property = {"PID": pid, "Name": name}
        shell_property = ansa.base.CreateEntity(ansa.constants.NASTRAN, "PSHELL", vals_property)
        return shell_property
    def set_pid(self, entity, pid):
        '''
        创建PID
        :param entity:      需创建的实体
        :param pid:         pid
        :return:
        '''
        ansa.base.SetEntityCardValues(ansa.constants.NASTRAN, entity, {"PID": pid})
    def new_project(self):
        ansa.session.New("discard")