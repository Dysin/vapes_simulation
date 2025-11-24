'''
@Desc:   路径管理
@Author: Dysin
@Date:   2025/10/24
'''

import os

class PathManager:
    """
    工程路径管理类
    用于统一管理项目路径结构：
    ├── data/
    ├── file/
    ├── geometry/
    ├── images/
    ├── mesh/
    └── simulation/
    """

    def __init__(self, root_dir: str, auto_create: bool = False):
        self.root_dir = os.path.abspath(root_dir)
        self.path_images = os.path.join(self.root_dir, "images")
        self.path_files = os.path.join(self.root_dir, "files")
        self.path_data = os.path.join(self.root_dir, "data")
        self.path_geometry = os.path.join(self.root_dir, "geometry")
        self.path_mesh = os.path.join(self.root_dir, "mesh")
        self.path_simulation = os.path.join(self.root_dir, "simulation")
        self.path_reports = os.path.join(self.root_dir, "reports")

        if auto_create:
            self._create_directories()

    def _create_directories(self):
        """自动创建所有目录"""
        for path in [
            self.root_dir,
            self.path_images,
            self.path_files,
            self.path_data,
            self.path_geometry,
            self.path_mesh,
            self.path_simulation,
            self.path_reports
        ]:
            os.makedirs(path, exist_ok=True)

    def get_subpath(self, base_path: str, *subpaths: str) -> str:
        """
        获取指定路径下的子路径
        示例：
            get_subpath(self.path_images, "result.png")
        """
        return os.path.join(base_path, *subpaths)

    def list_paths(self) -> dict:
        """返回所有路径字典"""
        return {
            "root": self.root_dir,
            "images": self.path_images,
            "files": self.path_files,
            "data": self.path_data,
            "simulation": self.path_simulation,
            "reports": self.path_reports
        }

    def __repr__(self):
        return f"<PathManager root={self.root_dir}>"