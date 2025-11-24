'''
@Desc:   输出网格文件
@Author: Dysin
@Date:   2025/10/28
'''

from ansa_utils import *

def output_mesh(path, project_name):
    ansa_utils = BaseUtils()
    ansa_utils.open_file(
        path = path,
        file_name = project_name,
        file_type = 'ansa'
    )
    mesh_utils = Mesh()
    mesh_utils.output(
        path = path,
        file_name = project_name + '_flow',
        type = 'nas'
    )
    pshells = Entities('PSHELL')
    pshell_outlet = pshells.get_entity(2)
    ansa.base.Or(pshell_outlet)
    mesh_utils.output(
        path=path,
        file_name=project_name + '_outlet',
        type='stl'
    )

if __name__ == '__main__':
    project_name = 'VP317E'
    path_root = r'D:\1_Work\active'
    path_mesh = os.path.join(path_root, 'project', project_name, 'mesh')
    output_mesh(path_mesh, project_name)