'''
@Desc:   获取几何参数
@Author: Dysin
@Date:   2025/10/17
'''

import os
import trimesh
import numpy as np
from scipy.spatial import ConvexHull, distance_matrix
from itertools import combinations

class STLUtils:
    def __init__(self, path, stl_name):
        self.path_stl = os.path.join(path, f'{stl_name}.stl')
        self.mesh = trimesh.load(self.path_stl)

    def get_bounding_box(self, stl_name):
        """
        读取STL文件并返回包围盒的最小和最大坐标值
        参数:
            stl_file_path (str): STL文件路径
        返回:
            dict: 包含最小和最大坐标值的字典
        """
        """使用 trimesh 库读取 STL 文件并获取包围盒"""
        try:
            # 获取包围盒
            min_coords, max_coords = self.mesh.bounds
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

    def get_area(self):
        '''
        获取STL总面积
        :param stl_name: STL名
        :return:
        '''
        print(f'[INFO] STL {self.path_stl} 面积为：{self.mesh.area} m^2')
        return self.mesh.area

    def get_points(self):
        """
        读取STL文件并返回所有顶点坐标
        """
        # 获取顶点坐标
        vertices = self.mesh.vertices
        faces = self.mesh.faces
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
        # 确保是三角网格
        if not isinstance(self.mesh, trimesh.Trimesh):
            print("错误: 加载的模型不是三角网格")
            return "UNKNOWN", None
        # 获取面法向量
        face_normals = self.mesh.face_normals
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

    def get_cylinder_diameter(self):
        '''
        根据圆柱STL计算其直径
        :param stl_name:
        :return:
        '''
        # 1. load mesh
        if self.mesh.is_empty:
            raise ValueError("加载的网格为空")
        # use vertex coordinates
        pts = self.mesh.vertices.copy()
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
        print(axis)
        return float(diameter), center

    def normalize_vector(self, vector):
        '''
        向量归一化
        :param vector: 向量
        :return:
        '''
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm

    def unique_vectors(self, vectors, tol=1e-6):
        """去重向量，包括反向向量"""
        unique = []
        for v in vectors:
            v = self.normalize_vector(v)
            # 检查是否已存在或反向存在
            exists = any(
                np.allclose(v, u, atol=tol) or
                np.allclose(v, -u, atol=tol)
                for u in unique
            )
            if not exists:
                unique.append(v)
        return unique

    def plane_normals_from_vectors(self, vectors):
        """计算任意两个向量所处平面的法向量"""
        plane_normals = []
        for v1, v2 in combinations(vectors, 2):
            n = np.cross(v1, v2)
            if np.linalg.norm(n) > 1e-8:  # 排除平行向量
                plane_normals.append(self.normalize_vector(n))
        return self.unique_vectors(plane_normals)

    def get_cylinder_aixs(self):
        normals = self.mesh.face_normals
        normals_unit = normals / np.linalg.norm(normals, axis=1)[:, None]

        # ------------------------------
        # 1. 尝试检测端面（大平面聚类）
        # ------------------------------
        angle_thr = np.deg2rad(5)
        groups = []
        used = np.zeros(len(normals), dtype=bool)

        for i, n in enumerate(normals_unit):
            if used[i]:
                continue
            dots = normals_unit @ n
            angles = np.arccos(np.clip(dots, -1, 1))
            idx = np.where(angles < angle_thr)[0]
            used[idx] = True
            groups.append(idx)

        # 过滤掉太小的组（侧面不会成大组）
        end_groups = [g for g in groups if len(g) > 30]

        # ------------------------------
        # 2. 若找到端面 → 直接输出法向量
        # ------------------------------
        if len(end_groups) >= 1:
            # 取最大的一组（通常就是端面）
            g = sorted(end_groups, key=lambda x: len(x), reverse=True)[0]
            n = np.mean(normals_unit[g], axis=0)
            n /= np.linalg.norm(n)
            return n

        # ------------------------------
        # 3. 若只有侧面 → 使用 PCA 推断轴向
        # ------------------------------
        cov = np.cov(normals_unit.T)
        eigvals, eigvecs = np.linalg.eigh(cov)

        # 法向量变化最小的方向 → 圆柱轴向
        axis = eigvecs[:, np.argmin(eigvals)]
        axis /= np.linalg.norm(axis)
        print(axis)
        return axis

    def oriented_bbox_along_axis(self, axis):
        """
        基于给定轴向的orientented bounding box (OBB)
        返回: center, extents(size), rotation_matrix, corner_points
        """
        # normalize axis
        axis = axis / np.linalg.norm(axis)
        # 1. 构造正交坐标系
        # 选择任意不平行向量
        tmp = np.array([1, 0, 0])
        if abs(np.dot(tmp, axis)) > 0.9:
            tmp = np.array([0, 1, 0])
        x = np.cross(tmp, axis)
        x /= np.linalg.norm(x)
        y = np.cross(axis, x)
        y /= np.linalg.norm(y)
        z = axis  # 轴向
        # 旋转矩阵：全局 → 局部
        R = np.vstack([x, y, z]).T  # (3,3)

        # 2. 点投影到局部坐标
        verts = self.mesh.vertices
        local = verts @ R  # shape (N,3)

        # 3. 求局部 AABB
        min_local = local.min(axis=0)
        max_local = local.max(axis=0)
        extents = max_local - min_local  # Lx, Ly, Lz
        # 包围盒中心（局部）
        center_local = (min_local + max_local) / 2

        # 4. 转回全局
        center_global = center_local @ R.T

        # 5. 计算 OBB 8 个角点（可选）
        dx, dy, dz = extents / 2
        corners_local = np.array([
            [dx, dy, dz],
            [dx, dy, -dz],
            [dx, -dy, dz],
            [dx, -dy, -dz],
            [-dx, dy, dz],
            [-dx, dy, -dz],
            [-dx, -dy, dz],
            [-dx, -dy, -dz],
        ])
        # 投影回全局
        corners_global = corners_local @ R.T + center_global
        return {
            "center": center_global,
            "extents": extents,  # 长宽高
            "rotation": R,  # 局部坐标系单位向量
            "corners": corners_global
        }

    def top_bottom_faces_from_obb(self, obb_dict, axis):
        """
        从 OBB 计算与轴平行的两面中心点和边长
        输入:
            obb_dict: oriented_bbox_along_axis 返回的字典
            axis: 圆柱轴向单位向量
        返回:
            dict: {
                "face1": {"center": ..., "edge_lengths": [...]},
                "face2": {"center": ..., "edge_lengths": [...]}
            }
        """
        corners = obb_dict["corners"]  # (8,3)
        R = obb_dict["rotation"]  # (3,3)
        extents = obb_dict["extents"]  # (3,)

        # 1. 找出哪个局部方向与 axis 平行
        axis = axis / np.linalg.norm(axis)
        dots = np.abs(R.T @ axis)  # 检查每个局部坐标轴与axis夹角
        idx = np.argmax(dots)  # 与 axis 最接近的局部轴索引
        # 对应的局部方向法向量
        face_normal = R[:, idx]

        # 2. 计算中心点
        # 角点索引按局部坐标系符号组合：
        # 局部 z 对应 idx
        # 面1: +方向, 面2: -方向
        # 8个角点的局部坐标: ±dx, ±dy, ±dz
        # 用 extents 来定位
        half_sizes = extents / 2

        # 面1中心点（axis + 方向）
        center1_local = np.zeros(3)
        center1_local[idx] = half_sizes[idx]
        center2_local = np.zeros(3)
        center2_local[idx] = -half_sizes[idx]

        # 投影回全局
        center1_global = center1_local @ R.T + obb_dict["center"]
        center2_global = center2_local @ R.T + obb_dict["center"]

        # 3. 面边长 = 其他两个方向 extents
        edge_axes = [0, 1, 2]
        edge_axes.remove(idx)
        edge_lengths1 = extents[edge_axes]
        edge_lengths2 = extents[edge_axes]

        return {
            "face1": {"center": center1_global, "edge_lengths": edge_lengths1},
            "face2": {"center": center2_global, "edge_lengths": edge_lengths2},
            "normal": face_normal
        }

    def get_cylinder_params(self):
        """
        获取圆柱几何参数
        返回:
            result = {
                "radius": float,     # 圆柱直径
                "center1": ndarray,  # 下端圆心
                "center2": ndarray,  # 上端圆心
                "axis": ndarray      # 单位轴向向量
            }
        """
        axis = self.get_cylinder_aixs()
        box = self.oriented_bbox_along_axis(axis)
        params = self.top_bottom_faces_from_obb(box, axis=axis)
        return {
            'center1': params['face1']['center'],
            'center2': params['face2']['center'],
            'radius': params['face1']['edge_lengths'][0] * 0.5,
            'axis': params['normal']
        }

if __name__ == '__main__':
    geo_utils = STLUtils(r'D:\1_Work\active\202510_ATN021\mesh')
    geo_params = geo_utils.get_bounding_box('ATN021_flow')
    for key, value in geo_params.items():
        print(f'{key}: {value}')