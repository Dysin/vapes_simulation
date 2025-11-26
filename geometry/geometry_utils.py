'''
@Desc:   获取几何参数
@Author: Dysin
@Date:   2025/10/17
'''

import os
import trimesh
import numpy as np
from scipy.spatial import ConvexHull, distance_matrix

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
            # print(f'[INFO] 几何包围盒尺寸：{box_params}')
        except Exception as e:
            print(f"处理 STL 文件时出错: {str(e)}")
            return None
        return box_params

    def get_stl_area(self, stl_name):
        '''
        获取STL总面积
        :param stl_name: STL名
        :return:
        '''
        stl_file = os.path.join(self.path, f'{stl_name}.stl')
        mesh = trimesh.load(stl_file, force='mesh')
        print(f'[INFO] STL {stl_name} 面积为：{mesh.area} m^2')
        return mesh.area

    def get_stl_points(self, stl_name):
        """
        读取STL文件并返回所有顶点坐标
        """
        stl_file = os.path.join(self.path, f'{stl_name}.stl')
        # 加载STL文件
        mesh = trimesh.load_mesh(stl_file)
        # 获取顶点坐标
        vertices = mesh.vertices
        faces = mesh.faces
        # print(f"顶点数量: {len(vertices)}")
        # print(f"面片数量: {len(faces)}")
        # 显示前几个顶点坐标
        # print("\n前10个顶点坐标:")
        # for i, vertex in enumerate(vertices[:10]):
        #     print(
        #     f"顶点 {i}: (
        #     {vertex[0]:.3f},
        #     {vertex[1]:.3f},
        #     {vertex[2]:.3f})"
        #     )
        return vertices

    def detect_plane_orientation(
            self,
            stl_name,
            threshold=0.95
    ):
        '''
        判断STL平面的法向方向
        :param stl_name: STL名
        :param threshold: 判断阈值(默认0.9)，值越大判断越严格
        :return:
            - 法向方向字符串('X', 'Y', 'Z'或'UNKNOWN')
            - 法向量
        '''
        # 加载STL文件
        stl_file = os.path.join(self.path, f'{stl_name}.stl')
        mesh = trimesh.load_mesh(stl_file)
        # 确保是三角网格
        if not isinstance(mesh, trimesh.Trimesh):
            print("错误: 加载的模型不是三角网格")
            return "UNKNOWN", None
        # 获取面法向量
        face_normals = mesh.face_normals
        # 计算法向量的平均值（对于平面，所有面法向量应该大致相同）
        avg_normal = np.mean(face_normals, axis=0)
        # 归一化平均法向量
        avg_normal = avg_normal / np.linalg.norm(avg_normal)
        # print(
        # f"平均法向量: (
        # {avg_normal[0]:.4f},
        # {avg_normal[1]:.4f},
        # {avg_normal[2]:.4f})"
        # )
        # 计算与各坐标轴方向的点积（余弦值）
        dot_x = abs(np.dot(avg_normal, [1, 0, 0]))  # 与X轴夹角余弦
        dot_y = abs(np.dot(avg_normal, [0, 1, 0]))  # 与Y轴夹角余弦
        dot_z = abs(np.dot(avg_normal, [0, 0, 1]))  # 与Z轴夹角余弦
        # 判断法向方向
        if dot_x > threshold and dot_x > dot_y and dot_x > dot_z:
            return "x", avg_normal
        elif dot_y > threshold and dot_y > dot_x and dot_y > dot_z:
            return "y", avg_normal
        elif dot_z > threshold and dot_z > dot_x and dot_z > dot_y:
            return "z", avg_normal
        else:
            return "UNKNOWN", avg_normal

    def get_cylinder_diameter(
            self,
            stl_name
    ):
        '''
        根据圆柱STL计算其直径
        :param stl_name:
        :return:
        '''
        # 1. load mesh
        stl_file = os.path.join(self.path, f'{stl_name}.stl')
        mesh = trimesh.load_mesh(stl_file, process=True)  # process 合并/修复
        if mesh.is_empty:
            raise ValueError("加载的网格为空")
        # use vertex coordinates
        pts = mesh.vertices.copy()
        # remove NaN/Inf and duplicates
        mask = np.isfinite(pts).all(axis=1)
        pts = pts[mask]
        if pts.shape[0] < 10:
            raise ValueError("顶点太少，无法计算")
        # optionally unique
        # round to avoid tiny numerical duplicates
        pts = np.unique(np.round(pts, 10), axis=0)
        # 2. PCA via SVD to find principal directions
        center = pts.mean(axis=0)
        X = pts - center
        # SVD
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        # Vt[0] is direction of largest variance -> cylinder axis
        axis = Vt[0]
        axis = axis / np.linalg.norm(axis)
        # 3. make two orthonormal vectors perpendicular to axis
        # pick arbitrary vector not parallel to axis
        arbitrary = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(arbitrary, axis)) > 0.9:
            arbitrary = np.array([0.0, 1.0, 0.0])
        u = np.cross(axis, arbitrary)
        u /= np.linalg.norm(u)
        v = np.cross(axis, u)
        v /= np.linalg.norm(v)
        # Project points into the plane spanned by u,v (2D coordinates)
        proj = np.vstack((pts.dot(u), pts.dot(v))).T  # shape (N,2)
        # 4. convex hull of projection
        try:
            hull = ConvexHull(proj)
            hull_pts = proj[hull.vertices]
        except Exception as e:
            # if convex hull fails (e.g., degenerate), fallback to all points
            hull_pts = proj
        # compute pairwise distances on hull points and take max -> diameter
        if hull_pts.shape[0] < 2:
            raise ValueError("投影点太少，无法计算直径")
        dists = distance_matrix(hull_pts, hull_pts)
        diameter = dists.max()
        # also return estimated axis and center for debugging
        return float(diameter), center

if __name__ == '__main__':
    geo_utils = GeometryUtils(r'D:\1_Work\active\202510_ATN021\mesh')
    geo_params = geo_utils.get_stl_bounding_box('ATN021_flow')
    for key, value in geo_params.items():
        print(f'{key}: {value}')