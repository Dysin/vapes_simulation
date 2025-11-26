'''
@Desc:   几何清理，GUI
@Author: Dysin
@Date:   2025/10/28
'''

from ansa_utils import *

@session.defbutton('GEOMETRY_PRE', 'geometry_process','几何清理')

def geometry_process():
	geometry = Geometry()
	geometry.delete_points_and_curves()
	ansa.base.Compress('')  # 清理
	ansa.base.Orient()
	mesh_utils = Mesh()
	mesh_utils.aspacing_stl(0.001, 0.01, 1)
	ansa.mesh.CreateStlMesh()
	ansa_utils = BaseUtils()
	ansa_utils.new_pid(1, 'inlet')
	ansa_utils.new_pid(2, 'outlet')
	ansa_utils.new_pid(3, 'airflow_sensor')
	ansa_utils.new_pid(4, 'wall')
	ansa_utils.new_pid(5, 'atomizer_core')
	print('[INFO] Finished!!!')