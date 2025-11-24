'''
@Desc:   生成网格
@Author: Dysin
@Date:   2024/8/4
'''

import math
from ansa_utils import *
import csv

class Mesh_Gen:
    def __init__(self, path, file_name):
        '''
        网格生成
        :param path:        文件根目录
        :param file_name:   文件名
        '''
        self.path = path
        self.file_name = file_name
        self.base_utils = BaseUtils()

    def wt_farfield(self, faces_infos):
        path_geo = os.path.join(self.path, 'geometry')
        path_mesh = os.path.join(self.path, 'mesh')
        self.base_utils.open_file(path_geo, self.file_name, file_type='stp')
        geometry = Geometry()
        geometry.delete_points_and_curves()
        faces = Entities('FACE')
        pshells = Entities('PSHELL')
        faces_all = faces.entities_all()
        ansa.base.Orient()  # orient
        for face_infos in faces_infos:
            self.base_utils.new_pid(face_infos[0], face_infos[1])
        for face in faces_all:
            self.base_utils.set_pid(face, faces_infos[-1][0])
        for face in faces_all:
            measurement = Measurement(face)
            area = measurement.entity_info()
            for face_infos in faces_infos:
                if area < face_infos[2] + 1000 and area > face_infos[2] - 1000:
                    self.base_utils.set_pid(face, face_infos[0])
        ansa.base.Compress('')      # 清理
        pshells_farfield = []
        pshells_wall = []
        pshells_farfield.append(pshells.get_entity(1))
        pshells_farfield.append(pshells.get_entity(2))
        pshells_farfield.append(pshells.get_entity(3))
        for i in range(3, len(faces_infos)):
            pshells_wall.append(pshells.get_entity(i + 1))
        ansa.base.Or(pshells_farfield)
        Mesh().aspacing_cfd(1000, 1000)
        ansa.mesh.CreateCfdMesh()
        ansa.base.Or(pshells_wall)
        Mesh().aspacing_cfd(5, 10)
        ansa.mesh.CreateCfdMesh()
        ansa.base.All()
        Mesh().generate_volume_mesh('HEXA INTERIOR')
        Mesh().output(path_mesh, self.file_name, type='Fluent')

    def wbt_farfield(self, first_layer_height, data, wingspan, refinement_size):
        print('WINGSPAN: ', wingspan)
        print('FIRST_LAYER: ', first_layer_height)
        path_geo = os.path.join(self.path, 'geometry')
        path_mesh = os.path.join(self.path, 'mesh')

        self.base_utils.open_file(path_geo, self.file_name, file_type='stp')
        geometry = Geometry()
        geometry.delete_points_and_curves()
        ansa.base.Orient()  # orient
        faces = Entities('FACE')
        pshells = Entities('PSHELL')
        Mesh().aspacing_stl(0.002, 0.05, 10)
        faces_all = faces.entities_all()
        # 创建PID
        for i in range(len(data)):
            self.base_utils.new_pid((i + 1), data[i][0])
        areas = []
        for face in faces_all:
            self.base_utils.set_pid(face, 5)
        for face in faces_all:
            measurement = Measurement(face)
            area = float(measurement.entity_info())
            areas.append(area)
            print(area)
            for group in data:
                for value in group[1:]:
                    if abs(float(value) * 1e6 - area) < 10 * math.log(area):
                        # print(abs(float(value) * 1e6 - area), math.log(area))
                        self.base_utils.set_pid(face, group[1])
        areas.sort(reverse=True)
        outlet_faceid = 0
        fuselage_faceid = 0
        for i in range(len(faces_all)):
            measurement = Measurement(faces_all[i])
            area = float(measurement.entity_info())
            if area == areas[0] or area == areas[1]:    # side
                self.base_utils.set_pid(faces_all[i], 7)
            elif area == areas[3] or area == areas[4]:  # inlet
                self.base_utils.set_pid(faces_all[i], 6)
            elif area == areas[2]:  # outlet
                outlet_faceid = i
            elif area == areas[5]:
                fuselage_faceid = i
        ansa.base.Compress('')  # 清理
        pshells_farfield = []
        pshells_wall = []
        pshells_farfield.append(pshells.get_entity(6))
        pshells_farfield.append(pshells.get_entity(7))
        pshells_farfield.append(pshells.get_entity(8))
        for i in range(5):
            pshells_wall.append(pshells.get_entity(i + 1))
        print(pshells_wall)

        # 判断Orient是否错误，反转法向
        ansa.base.Or(pshells_wall)
        fuselage_normal = faces.get_face_orientation(fuselage_faceid)
        if fuselage_normal[2] > 0:
            ansa.base.Orient()
        ansa.base.Or(pshells_farfield)
        outlet_normal = faces.get_face_orientation(outlet_faceid)
        if outlet_normal[0] > 0:
            ansa.base.Orient()  # orient

        Mesh().aspacing_cfd(1000, 1000)
        ansa.mesh.CreateCfdMesh()
        ansa.base.Or(pshells_wall)
        Mesh().aspacing_cfd(4, 8)
        ansa.mesh.CreateCfdMesh()
        ansa.base.All()
        # 网格加密
        min_coords = [
            -wingspan * 0.5,
            -wingspan * 1.1,
            -wingspan * 0.2
        ]
        max_coords = [
            wingspan * 3,
            wingspan * 1.1,
            wingspan * 0.8
        ]
        Mesh().create_mesh_refinement(
            min_coords,
            max_coords,
            refinement_size,
            refinement_size
        )
        # 生成边界层
        Mesh().generate_boundary_layers(pshells_wall, len(data) + 1, first_layer_height, 1.3, 15)
        # 生成体网格
        Mesh().generate_volume_mesh('HEXA INTERIOR')
        Mesh().output(path_mesh, self.file_name, type='Fluent')

if __name__ == '__main__':
    path_root = r'F:\05_Special_Projects\202406_aircraft_project\cfd\03_ZQ10'

    file_mesh_params = os.path.join(path_root, 'geometry', 'mesh_params.csv')
    file = open(file_mesh_params, mode='r')
    mesh_params = list(csv.reader(file))
    file.close()
    geom_name = mesh_params[2][1]
    star_num = int(mesh_params[3][1])
    end_num = int(mesh_params[4][1])
    print(geom_name, star_num, end_num)

    for i in range(star_num, end_num):
        BaseUtils().new_project()
        if star_num == -1:
            file_name = geom_name
        else:
            file_name = geom_name + '_' + '{:03d}'.format(i+1)
        file_csv = os.path.join(path_root, 'geometry', file_name + '.csv')
        file_data = open(file_csv, mode='r')
        data = list(csv.reader(file_data))
        file_data.close()

        mesh_gen = Mesh_Gen(path_root, file_name)
        # faces_infos = [
        #     [1, 'inlet', 1.15628958667E9],
        #     [2, 'side', 1.734626856091375E9],
        #     [3, 'outlet', 1156445279],
        #     [4, 'wing_upper', 204848],
        #     [5, 'wing_lower', 199119],
        #     [6, 'wall', -1]
        # ]
        # mesh_gen.wt_farfield(faces_infos)

        wingspan = float(mesh_params[0][1]) * 1000     # [mm]
        first_layer_height = float(mesh_params[1][1]) * 1000
        refinement_size = 200
        mesh_gen.wbt_farfield(first_layer_height, data, wingspan, refinement_size)