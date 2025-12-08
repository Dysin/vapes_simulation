'''
@Desc:   输出优化区域网格
@Author: Dysin
@Date:   2025/12/8
'''

def output_opt_region_mesh(path, mesh_name, region_num):
    ansa_utils = BaseUtils()
    ansa_utils.input_mesh(
        path=path,
        file_name=mesh_name,
        file_type='nas'
    )
    mesh_utils = Mesh()
    pshells = Entities('PSHELL')
    pshells_all = pshells.entities_all()

    for pshell in pshells_all:
        print(pshell._name)
        for i in range(region_num):
            region_name = 'opt_region' + str(i+1)
            if pshell._name == region_name:
                # pshell = pshells.get_entity(pshell._id)
                ansa.base.Or(pshell)
                mesh_utils.output(
                    path=path,
                    file_name=mesh_name + '_' + region_name,
                    type='stl'
                )
                break

if __name__ == '__main__':
    mesh_name = '20251201_airway'
    path_mesh = r'E:\1_Work\active\airway_analysis\VP353\mesh\doe'
    output_opt_region_mesh(path_mesh, mesh_name, 2)
