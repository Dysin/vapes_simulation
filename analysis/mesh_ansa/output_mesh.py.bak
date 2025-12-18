'''
@Desc:   输出网格文件
@Author: Dysin
@Date:   2025/10/28
'''

from ansa_utils import *

def output_mesh(path, mesh_name):
    ansa_utils = BaseUtils()
    ansa_utils.input_mesh(
        path = path,
        file_name = mesh_name,
        file_type = 'nas'
    )
    # ansa_utils.open_file(
    #     path = path,
    #     file_name = mesh_name
    # )
    mesh_utils = Mesh()
    # mesh_utils.output(
    #     path = path,
    #     file_name = mesh_name,
    #     type = 'nas'
    # )
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

if __name__ == '__main__':
    mesh_name = '20251218_airway'
    path_mesh = r'E:\1_Work\active\airway_analysis\VP158A\mesh\origin'
    output_mesh(path_mesh, mesh_name)
