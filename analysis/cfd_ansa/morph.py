'''
@Desc:   
@Author: Dysin
@Date:   2025/11/6
'''

import csv
import numpy as np
from ansa_utils import *

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
        morph_utils.morph_cylindrical(i+1, radius=radius_target[i])
    mesh_name = ansa_name + '_b' + csv_id + '_' + "{:02d}".format(name_id)
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
    outlet_id = 2
    for pshell in pshells_all:
        print(pshell._name)
        if pshell._name == 'outlet':
            outlet_id = pshell._id
    pshell_outlet = pshells.get_entity(outlet_id)
    ansa.base.Or(pshell_outlet)
    mesh_utils.output(
        path=path,
        file_name=mesh_name + '_outlet',
        type='stl'
    )


if __name__ == '__main__':
    proj_name = 'VP353'
    ver_number = '20251103'
    path_root = r'D:\1_Work\active\project'
    path_data = os.path.join(path_root, proj_name, 'data')
    path_opt = os.path.join(path_root, proj_name, 'mesh', 'doe')
    csv_id = '02'
    nas_name = proj_name + '_' + ver_number + '_spf'
    csv_input = os.path.join(
        path_data,
        proj_name + '_' + ver_number + '_input_params_' + csv_id + '.csv'
    )
    file_data = open(csv_input, mode='r')
    df = list(csv.reader(file_data))
    file_data.close()

    print('[INFO] Mesh name: ', nas_name)
    points = [
        # [[-64.1, 0, 0], [-86.9, 0, 0]],
        [[-45.5, -5.7, 0.33], [-10., -5.7, 0.33]],
        [[-0.3, -5.7, 0.33], [-1.6, -5.7, 0.33]],
    ]
    radius = [
        # 2.3,
        1.27,
        1.16
    ]

    for i in range(1, len(df)):
        delta_r = np.array(df[i]).astype(float)
        mesh_morph(path_opt, nas_name, points, radius, delta_r, i, csv_id)