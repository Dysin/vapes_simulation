'''
@Desc:   图片管理器
@Author: Dysin
@Date:   2025/11/24
'''

import os
import numpy as np
from PIL import Image, ImageOps
import matplotlib
matplotlib.use("TkAgg")   # 或 "Qt5Agg"
import matplotlib.pyplot as plt

class ImageUtils():
    def add_black_border(self, img, border_size):
        """
        Adds a black border around the image.
        :param img: The original PIL Image object.
        :param border_size: Size of the border (default is 10 pixels).
        :return: A new PIL Image object with the black border added.
        """
        return ImageOps.expand(img, border=border_size, fill='black')

    def bmp_to_png(
            self,
            path_bmp,
            path_png,
            image_name,
            compress_level=0,
            add_border=False,
            border_size=2
    ):
        image_bmp = os.path.join(path_bmp, f'{image_name}.bmp')
        image_png = os.path.join(path_png, f'{image_name}.png')

        # 打开 BMP 文件
        with Image.open(image_bmp) as img:
            # 如果需要加黑色边框
            if add_border:
                img = self.add_black_border(img, border_size)

            # 保存为 PNG 格式，调整压缩级别
            img.save(
                image_png,
                'PNG',
                optimize=True,
                compress_level=compress_level
            )
        print(f'[INFO] 已将图片 {image_bmp} 转换为 {image_png}')

class PltImage3D:
    def __init__(
            self,
            path,
            image_name,
            figsize=(10, 9),
            fontsize=10,
    ):
        self.fig = plt.figure(figsize=figsize)
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.fontsize = fontsize
        self.path_img = os.path.join(path, f'{image_name}.png')

    def merge_to_3cols(self, col2, col1, col0=None):
        '''
        将 n×2 和 n×1 的数据合并为 n×3
        :param col2: list | np.ndarray | pd.DataFrame，n×2 数据
        :param col1: list | np.ndarray | pd.Series | pd.DataFrame，n×1 数据
        :return: np.ndarray，n×3 numpy 数组
        '''

        # ---------- 转为 numpy ----------
        col2 = np.asarray(col2)
        col1 = np.asarray(col1)

        # ---------- 维度检查 ----------
        if col0 is None:
            if col2.ndim != 2 or col2.shape[1] != 2:
                raise ValueError(f"col2 必须是 n×2，当前 shape={col2.shape}")
            # col1 允许 (n,), (n,1)
            if col1.ndim == 1:
                col1 = col1.reshape(-1, 1)
            if col1.ndim != 2 or col1.shape[1] != 1:
                raise ValueError(f"col1 必须是 n×1，当前 shape={col1.shape}")

        # ---------- 行数检查 ----------
        if col2.shape[0] != col1.shape[0]:
            raise ValueError(
                f"行数不一致: col2={col2.shape[0]}, col1={col1.shape[0]}"
            )

        if col0 is None:
            res = np.hstack((col2, col1))
        else:
            res = np.column_stack([col2, col1, col0])

        # ---------- 合并 ----------
        return res

    def scatters_and_curves(self, scatters_points, curves_points):
        '''
        绘制三维曲线，如B样条曲线
        :param scatters_points: 点集
        :param curves_points:   曲线点集
        :return:
        '''
        scatters_points_x = scatters_points[:, 0]
        scatters_points_y = scatters_points[:, 1]
        scatters_points_z = scatters_points[:, 2]
        curves_points_x = curves_points[:, 0]
        curves_points_y = curves_points[:, 1]
        curves_points_z = curves_points[:, 2]
        # 绘制曲线
        self.ax.plot(
            curves_points_x,
            curves_points_y,
            curves_points_z,
            label='Curves'
        )
        # 绘制点
        self.ax.scatter(
            scatters_points_x,
            scatters_points_y,
            scatters_points_z,
            color='red',
            label='Points'
        )

        # x、y 和 z 轴等比例显示
        max_range = np.array([
            curves_points_x.max() - curves_points_x.min(),
            curves_points_y.max() - curves_points_y.min(),
            curves_points_z.max() - curves_points_z.min()
        ]).max()
        mid_x = (curves_points_x.max() + curves_points_x.min()) * 0.5
        mid_y = (curves_points_y.max() + curves_points_y.min()) * 0.5
        mid_z = (curves_points_z.max() + curves_points_z.min()) * 0.5
        self.ax.set_xlim(mid_x - max_range / 2, mid_x + max_range / 2)
        self.ax.set_ylim(mid_y - max_range / 2, mid_y + max_range / 2)
        self.ax.set_zlim(mid_z - max_range / 2, mid_z + max_range / 2)

        self.ax.legend()
        # self.ax.set_title('3D B-spline with endpoint slopes (Y=0)')
        # 设置轴的等间距显示
        self.ax.set_box_aspect([1, 1, 1])  # x, y, z 轴的比例相同
        plt.show()

    def scatters_and_surface(
            self,
            scatters_x,
            scatters_y,
            scatters_z,
            grid_x=None,
            grid_y=None,
            grid_z=None,
            points_size=20,
            text_position=None,
            text=None,
            axis_labels=None,
            title=None
    ):
        if axis_labels is None:
            self.ax.set_xlabel('x')
            self.ax.set_ylabel('y')
            self.ax.set_zlabel('z')
        else:
            self.ax.set_xlabel(axis_labels[0])
            self.ax.set_ylabel(axis_labels[1])
            self.ax.set_zlabel(axis_labels[2])
        # 绘制点
        self.ax.scatter(
            scatters_x,
            scatters_y,
            scatters_z,
            color='red',
            s=points_size,
            label='Points'
        )
        if grid_x is not None:
            # 绘制三维曲面
            surf = self.ax.plot_surface(
                grid_x,
                grid_y,
                grid_z,
                cmap='viridis',     # 颜色映射方案：viridis/plasma
                alpha=0.9,          # 曲面透明度
                edgecolor='none',   # 定义网格线颜色，'none' 表示没有网格线
                linewidth = 0,      # 网格线宽（0 表示不显示网格线）
            )
            # 颜色条
            cbar = self.fig.colorbar(
                surf,
                ax=self.ax,
                shrink=0.2,         # 颜色条长度比例（相对于坐标轴）
                aspect=16,          # 长宽比，数值越大越“细长”
                pad=0.06             # 颜色条与图像间距
            )
            # 颜色条标签
            # cbar.set_label(
            #     'Z value',
            #     fontsize=12,
            #     labelpad=10  # 标签与颜色条的间距
            # )
            # 颜色条刻度字号
            cbar.ax.tick_params(
                labelsize=self.fontsize,  # 刻度字号
                width=1.0,  # 刻度线宽
                length=4  # 刻度线长度
            )
        # 坐标轴标签与标题设置
        if title is not None:
            self.ax.set_title(
                title,
                fontsize = self.fontsize,
                pad=15  # 标题与图的间距
            )
        # 添加文本
        if text_position is not None:
            self.ax.text(
                text_position[0],
                text_position[1],
                text_position[2],
                text,
                color='black',
                fontsize=self.fontsize
            )
        # 坐标轴刻度风格
        self.ax.tick_params(
            axis='both',
            which='major',
            labelsize=self.fontsize,  # 主刻度字号
            width=1.0,
            length=5
        )

        # z 轴刻度单独设置（3D 图常用）
        self.ax.zaxis.set_tick_params(
            labelsize=self.fontsize,
            width=1.0,
            length=5
        )
        # 视角设置（非常重要）
        self.ax.view_init(
            elev=25,  # 仰角（上下）
            azim=135  # 方位角（左右）
        )
        # plt.show()
        print(self.path_img)
        plt.subplots_adjust(
            left=0.05,
            right=1.02,  # 右边几乎贴边
            bottom=0.01,
            top=0.99
        )
        plt.savefig(self.path_img, dpi=500)

if __name__ == '__main__':
    image = ImageUtils()
    path_bmp = r'D:\1_Work\active\202510_VP322-B\simulation\steady_rans_flow'
    path_png = r'D:\1_Work\active\202510_VP322-B\file\result_analysis_flow'
    image_name = 'ff'
    image.bmp_to_png(path_bmp, path_png, image_name, compress_level=1)
