'''
@Desc:   获取几何参数
@Author: Dysin
@Date:   2025/10/17
'''

import os
import trimesh

class GeometryUtils:
    def __init__(self, path):
        self.path = path

    def get_stl_bounding_box(self, stl_name):
        """
        读取STL文件并返回包围盒的最小和最大坐标值
        参数:
            stl_file_path (str): STL文件路径
        返回:
            dict: 包含最小和最大坐标值的字典
        """
        stl_file = os.path.join(self.path, f'{stl_name}.stl')
        # 加载STL文件
        """使用 trimesh 库读取 STL 文件并获取包围盒"""
        try:
            # 加载 STL 文件
            mesh = trimesh.load(stl_file)
            # 获取包围盒
            min_coords, max_coords = mesh.bounds
            box_params = {
                'min_x': min_coords[0],
                'min_y': min_coords[1],
                'min_z': min_coords[2],
                'max_x': max_coords[0],
                'max_y': max_coords[1],
                'max_z': max_coords[2],
                'size_x': max_coords[0] - min_coords[0],
                'size_y': max_coords[1] - min_coords[1],
                'size_z': max_coords[2] - min_coords[2],
            }
            print(f'[INFO] 几何包围盒尺寸：{box_params}')
        except Exception as e:
            print(f"处理 STL 文件时出错: {str(e)}")
            return None
        return box_params

if __name__ == '__main__':
    geo_utils = GeometryUtils(r'D:\1_Work\active\202510_ATN021\mesh')
    geo_params = geo_utils.get_stl_bounding_box('ATN021_flow')
    for key, value in geo_params.items():
        print(f'{key}: {value}')