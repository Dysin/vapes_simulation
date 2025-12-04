'''
@Desc:   
@Author: Dysin
@Date:   2025/11/6
'''

import csv
import numpy as np
from ansa_utils import *

def output_mesh(path, mesh_name):
    mesh_utils = Mesh()
    mesh_utils.output(
        path=path,
        file_name=mesh_name,
        type='nas'
    )
    mesh_utils.output(
        path=path,
        file_name=mesh_name,
        type='stl'
    )
    pshells = Entities('PSHELL')
    pshells_all = pshells.entities_all()

    for pshell in pshells_all:
        print(pshell._name)
        if pshell._name == 'outlet':
            pshell = pshells.get_entity(pshell._id)
            ansa.base.Or(pshell)
            mesh_utils.output(
                path=path,
                file_name=mesh_name + '_outlet',
                type='stl'
            )
        elif pshell._name == 'inlet':
            pshell = pshells.get_entity(pshell._id)
            ansa.base.Or(pshell)
            mesh_utils.output(
                path=path,
                file_name=mesh_name + '_inlet',
                type='stl'
            )
        elif pshell._name == 'atomizer_core':
            pshell = pshells.get_entity(pshell._id)
            ansa.base.Or(pshell)
            mesh_utils.output(
                path=path,
                file_name=mesh_name + '_atomizer_core',
                type='stl'
            )

def mesh_morph(
        path,
        ansa_name,
        points,
        radius_orig,
        radius_target,
        name_id,
        csv_id
):
    geo_utils = Geometry()
    base_utils = BaseUtils()
    morph_utils = MorphUtils()
    mesh_utils = Mesh()
    base_utils.new_project()
    base_utils.input_mesh(path, ansa_name, 'nas')
    for i in range(len(radius_orig)):
        curve = geo_utils.create_line(points[i][0], points[i][1])
        morph_utils.morph_status(True)
        morph = morph_utils.create_cylindrical_box(
            curve,
            radius1=radius_orig[i],
            radius2=radius_orig[i]
        )
        # morph_utils.morph_cylindrical(i+1, radius=radius_target[i])
    mesh_name = ansa_name + '_b' + csv_id + '_p' + "{:02d}".format(name_id)
    # output_mesh(path, mesh_name)

if __name__ == '__main__':
    proj_name = 'VP353'
    ver_number = '20251201'
    path_root = r'E:\1_Work\active\airway_analysis'
    path_data = os.path.join(path_root, proj_name, 'data')
    path_opt = os.path.join(path_root, proj_name, 'mesh', 'doe')
    csv_id = '01'
    nas_name = ver_number + '_airway'
    csv_input = os.path.join(
        path_data,
        ver_number + '_input_params_' + csv_id + '.csv'
    )
    file_data = open(csv_input, mode='r')
    df = list(csv.reader(file_data))
    file_data.close()

    print('[INFO] Mesh name: ', nas_name)
    points = [
        [[95.58, -8.2, 0.3346], [95.58, -7.28, 0.3346]],
        [[100.05, -6.14, 0.3349], [98.33, -6.14, 0.3349]],
        [[86.83, -6.16, 0.3307], [85.91, -6.16, 0.3307]],
    ]
    radius = [
        0.8,
        1.35,
        1.2
    ]

    for i in range(10, len(df)):
        delta_r = np.array(df[i]).astype(float)
        mesh_morph(path_opt, nas_name, points, radius, delta_r, i, csv_id)