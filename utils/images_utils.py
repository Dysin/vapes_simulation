'''
@Desc:   图片管理器
@Author: Dysin
@Date:   2025/11/24
'''

import os
import numpy as np
import pandas as pd
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

class PlotImage2D:
    # path_save:        保存图片路径
    # file_name:        图片名
    # colormap_name:    色系名称（viridis, plasma, inferno, magma, cividis）
    def __init__(
            self,
            path_save,
            file_name,
            label_x='x',
            label_y='y',
            colormap_name='1',
            font_size=22,
            figure_size=(12, 9)
    ):
        self.path_save = path_save
        self.file_name = file_name
        self.colormap_name = colormap_name
        self.label_x = label_x
        self.label_y = label_y
        self.path_image = os.path.join(path_save, f'{file_name}.png')
        # 设置全局字体大小
        self.font_size = font_size
        self.figure_size = figure_size
        # 设置xtick和ytick的方向：in、out、inout
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'
        # X、Y轴标签字体大小
        plt.rcParams['xtick.labelsize'] = self.font_size
        plt.rcParams['ytick.labelsize'] = self.font_size
        # X、Y轴刻度标签字体大小
        plt.rcParams['axes.labelsize'] = self.font_size
        # 设置默认字体为中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei']
        # 解决负号显示问题
        plt.rcParams['axes.unicode_minus'] = False
        # 设置图片长宽比例
        plt.figure(figsize=figure_size)

    def to_numpy(self, data):
        """
        自动识别类型并转为 numpy.ndarray。
        支持类型：
        - numpy.ndarray
        - pandas DataFrame
        - pandas Series
        - list / tuple
        - 标量（int/float）
        """
        if isinstance(data, np.ndarray):
            return data
        if isinstance(data, (pd.DataFrame, pd.Series)):
            return data.to_numpy()
        if isinstance(data, (list, tuple)):
            return np.array(data)
        if np.isscalar(data):
            return np.array([data])
        raise TypeError(f"不支持的数据类型：{type(data)}")

    def get_colormap(self):
        '''
        色系
        :return:
        '''
        colormaps = {
            '1': plt.cm.viridis,
            '2': plt.cm.plasma,
            '3': plt.cm.inferno,
            '4': plt.cm.magma,
            '5': plt.cm.cividis,
            # 添加更多的色系
        }
        return colormaps.get(self.colormap_name)  # 默认返回viridis色系
    def colors(self, num_colors):
        results = self.get_colormap()(np.linspace(0, 1, num_colors))
        return results
    def plt_curve(
            self,
            x,
            y,
            labels=None,
            range_x=None,
            range_y=None
    ):
        x = self.to_numpy(x)
        y = self.to_numpy(y)
        plt.plot(
            x,
            y,
            'k-',
            markersize = 2.5,
            linewidth = 2,
            label = labels
        )
        plt.legend(loc='best',frameon=False)    # 去掉图例边框，设置字体
        if range_x is None:
            plt.xlim(xmin=min(x), xmax=max(x))
        else:
            plt.xlim(xmin=range_x[0], xmax=range_x[1])
        if range_y is not None:
            plt.ylim(ymin=range_y[0], ymax=range_y[1])
        plt.grid(linestyle='--')  # 添加网格线
        plt.xlabel(self.label_x)
        plt.ylabel(self.label_y)
        plt.savefig(
            # 文件路径+文件名
            self.path_image,
            # @dpi 每英寸像素数，可以理解为清晰度或细腻度
            dpi = 1000
        )
        plt.close()
    def plt_multicurve(
            self,
            x,
            y,
            labels=None,
            legend_text=None,
            legend_size=None,
            range_x=None,
            range_y=None,
            invert_yaxis=False,
    ):
        x = self.to_numpy(x)
        y = self.to_numpy(y)
        colors = self.colors(len(y))
        if labels is None:
            for i in range(len(y)):
                plt.plot(
                    x[i],
                    y[i],
                    # self.styles[i],
                    color=colors[i],
                    markersize = 6.5,
                    linewidth = 2,
                )
        else:
            for i in range(len(y)):
                plt.plot(
                    x[i],
                    y[i],
                    # self.styles[i],
                    color=colors[i],
                    markersize = 6.5,
                    linewidth = 2,
                    label = labels[i]
                )
        if legend_text is not None:
            plt.legend(
                labels=legend_text,
                loc='best',
                frameon=False,
                fontsize=legend_size
            )    # 去掉图例边框，设置字体
        plt.xlabel(self.label_x)
        plt.ylabel(self.label_y)
        if range_x is None:
            plt.xlim(xmin=min(x[0]), xmax=max(x[0]))
        else:
            plt.xlim(xmin=range_x[0], xmax=range_x[1])
        if range_y is not None:
            plt.ylim(ymin=range_y[0], ymax=range_y[1])
        plt.grid(linestyle = '--')  # 添加网格线
        if invert_yaxis:
            plt.gca().invert_yaxis()
        plt.savefig(
            # 文件路径+文件名
            self.path_image,
            # @dpi 每英寸像素数，可以理解为清晰度或细腻度
            dpi = 600
        )
        plt.close()

    def plt_curve_with_points(
            self,
            scatter_x,
            scatter_y,
            curve_x,
            curve_y,
            markers=None,
            legend_text=None,
            legend_size=None,
            range_x=None,
            range_y=None,
            invert_yaxis=False,
            marker_size=8,
            line_width=3,
            colors=None
    ):
        """
        绘制曲线 + 点
        markers: 点样式列表，比如 ['o','s','^','d']，若 None 则自动循环
        """
        if colors is None:
            colors = self.colors(len(curve_y))
        # 默认点样式循环
        default_markers = ['o', 's', '^', 'd', 'v', 'p', '*', 'x']
        default_colors = ['k', 'g', 'r', 'c', 'm', 'y', 'b']
        if markers is None:
            markers = [
                default_markers[i % len(default_markers)]
                for i in range(len(curve_y))
            ]

        for i in range(len(curve_y)):
            print(curve_y[i])
            # 绘制曲线
            plt.plot(
                curve_x[i],
                curve_y[i],
                color=colors[i],
                linewidth=line_width
            )
        for i in range(len(scatter_y)):
            # 绘制点
            plt.scatter(
                scatter_x[i],
                scatter_y[i],
                color=default_colors[i],
                marker=markers[i],
                s=marker_size ** 2,   # s 为面积
            )

        if legend_text is not None:
            plt.legend(
                labels=legend_text,
                loc='best',
                frameon=False,
                fontsize=legend_size
            )

        plt.xlabel(self.label_x)
        plt.ylabel(self.label_y)

        # X 轴范围
        if range_x is None:
            plt.xlim(xmin=min(curve_x[0]), xmax=max(curve_x[0]))
        else:
            plt.xlim(xmin=range_x[0], xmax=range_x[1])

        # Y 轴范围
        if range_y is not None:
            plt.ylim(ymin=range_y[0], ymax=range_y[1])

        plt.grid(linestyle='--')

        # Y 轴反转
        if invert_yaxis:
            plt.gca().invert_yaxis()

        # 保存图片
        plt.savefig(
            self.path_image,
            dpi=600
        )
        plt.close()


    def bar_with_text(self, text, text_size, text_position):
        '''
        绘制单个柱状图
        :param text:            文本内容
        :param text_size:       文本字体大小
        :param text_position:   文本相对位置，如：[0.2, 0.1]
        :return:
        '''
        plt.bar(self.x, self.y, color='skyblue')
        plt.xlabel('%s' % self.label_x)
        plt.ylabel('%s' % self.label_y)
        # 计算文本位置（使用中间的x坐标和y轴的上限）
        # 获取当前坐标轴的范围
        xlim = plt.xlim()
        ylim = plt.ylim()
        # 添加文本
        if text is not None:
            # 设置文本位置为右上角
            text_x = xlim[1] - text_position[0] * (xlim[1] - xlim[0])
            text_y = ylim[1] - text_position[1] * (ylim[1] - ylim[0])
            plt.text(
                text_x,
                text_y,
                text,
                fontsize=text_size,
                ha='left',
                va='top'
            )
        plt.savefig(
            # 文件路径+文件名
            os.path.join(self.path_save, '%s.png' % self.file_name),
            # @dpi 每英寸像素数，可以理解为清晰度或细腻度
            dpi=600
        )
        plt.close()

    def bar(
            self,
            x,
            y,
            title=None,
            colors=None,
            grid_axis='both'
    ):
        """
        绘制双柱状图
        :param x: list, X轴标签
        :param y: list, Y轴数值
        :param title: str, 标题
        :param xlabel: str, X轴说明
        :param ylabel: str, Y轴说明
        :param colors: list or str, 柱子颜色
        :param grid_axis: str, 网格线方向 ('x', 'y', 'both')
        :param save_path: str, 如果提供路径则保存图片
        """
        x = self.to_numpy(x)
        y = self.to_numpy(y)
        # 1. 创建画布
        fig, ax = plt.subplots(figsize=(7, 5))

        # 2. 默认颜色设置
        if colors is None:
            colors = ['skyblue', 'lightcoral']

        # 3. 绘制柱状图
        # zorder=3 让柱子显示在网格线之上
        bars = ax.bar(x, y, color=colors, width=0.5, zorder=3)

        # 4. 设置网格线
        # zorder=0 让网格线在底层
        ax.grid(True, axis=grid_axis, linestyle=':', alpha=0.5, zorder=0)

        # 5. 添加数值标签 (在柱子顶部显示数字)
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f'{height}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),  # 向上偏移3个点
                textcoords="offset points",
                ha='center',
                va='bottom'
            )
        # 6. 装饰细节
        if title is not None:
            ax.set_title(title, fontsize=self.font_size, pad=15)
        ax.set_xlabel(self.label_x)
        ax.set_ylabel(self.label_y)
        # 7. 保存或显示
        plt.savefig(self.path_image, dpi=600)
        plt.close()
        print(f"图表已保存至: {self.path_image}")

    def grouped_bar(
            self,
            categories,
            data_dict,
            title=None,
            colors=None,
            width=0.35,
            show_values=True
    ):
        """
        绘制多组对比柱状图
        :param categories: X轴的类别标签列表，例如 ['A', 'B', 'C']
        :param data_dict: 字典格式，Key为组名，Value为对应类别的数值列表。
                          例如 {'Group1': [10, 20, 30], 'Group2': [15, 25, 35]}
        :param title: 图表标题
        :param colors: 颜色列表
        :param width: 单个柱子的宽度
        :param show_values: 是否在柱子顶部显示具体数值
        """

        # 1. 准备位置参数
        x = np.arange(len(categories))  # 基础刻度位置 [0, 1, 2...]
        n_groups = len(data_dict)  # 共有几组数据

        # 计算每组柱子的起始偏移，确保它们居中对齐
        # 如果有2组，偏移分别是 -width/2, +width/2
        # 如果有3组，偏移分别是 -width, 0, +width
        offsets = np.linspace(
            -width * (n_groups - 1) / 2,
            width * (n_groups - 1) / 2,
            n_groups
        )

        fig, ax = plt.subplots()

        # 2. 默认颜色
        if colors is None:
            colors = plt.cm.Paired(
                np.linspace(0, 1, n_groups)
            )

        # 3. 循环绘制每一组
        for i, (group_name, values) in enumerate(data_dict.items()):
            rects = ax.bar(
                x + offsets[i],
                values,
                width,
                label=group_name,
                color=colors[i],
                zorder=3
            )

            # 是否显示数值标签
            if show_values:
                for rect in rects:
                    height = rect.get_height()
                    ax.annotate(
                        f'{height}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center',
                        va='bottom',
                        fontsize=9
                    )

        # 4. 设置网格线 (zorder=0 确保网格在背景)
        ax.grid(True, axis='both', linestyle='--', alpha=0.8, zorder=0)

        # 5. 设置坐标轴和标签
        if title is not None:
            ax.set_title(title, fontsize=15, pad=20)
        ax.set_xlabel(self.label_x)
        ax.set_ylabel(self.label_y)
        ax.set_xticks(x)
        ax.set_xticklabels(categories)

        # 6. 图例
        ax.legend()

        # 自动调整布局，防止元素重叠
        plt.tight_layout()
        plt.savefig(self.path_image, dpi=600)
        plt.close()

class PlotImage3D:
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
                cmap='plasma',      # 颜色映射方案：viridis/plasma
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
