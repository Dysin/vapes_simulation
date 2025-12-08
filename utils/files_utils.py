'''
图片、视频库
Author: Dysin
Time:   2024.06.16
'''

import os
import csv
import glob
import shutil
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class Image:
    # path_save:        保存图片路径
    # file_name:        图片名
    # x,y:              x,y坐标数据，一维或二维数组
    # colormap_name:    色系名称（viridis, plasma, inferno, magma, cividis）
    def __init__(
            self,
            path_save,
            file_name,
            x,
            y,
            colormap_name='1',
            labels=None,
            label_x='x',
            label_y='y',
            size=22,
            figure_size=(12, 9),
    ):
        self.path_save = path_save
        self.file_name = file_name
        self.x = x
        self.y = y
        self.colormap_name = colormap_name
        self.labels = labels
        self.label_x = label_x
        self.label_y = label_y
        # 设置全局字体大小
        self.size = size
        # 设置xtick和ytick的方向：in、out、inout
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'
        # X、Y轴标签字体大小
        plt.rcParams['xtick.labelsize'] = self.size
        plt.rcParams['ytick.labelsize'] = self.size
        # X、Y轴刻度标签字体大小
        plt.rcParams['axes.labelsize'] = self.size
        # 设置默认字体为中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei']
        # 解决负号显示问题
        plt.rcParams['axes.unicode_minus'] = False
        # 设置图片长宽比例
        plt.figure(figsize=figure_size)

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
    def plt_curve(self, x_range=None, y_range=None):
        plt.plot(
            self.x,
            self.y,
            'k-',
            markersize = 2.5,
            linewidth = 2,
            label = self.labels
        )
        plt.legend(loc='best',frameon=False)    # 去掉图例边框，设置字体
        if x_range is None:
            plt.xlim(xmin=min(self.x), xmax=max(self.x))
        else:
            plt.xlim(xmin=x_range[0], xmax=x_range[1])
        if y_range is not None:
            plt.ylim(ymin=y_range[0], ymax=y_range[1])
        plt.grid(linestyle='--')  # 添加网格线
        plt.xlabel('%s' %self.label_x)
        plt.ylabel('%s' %self.label_y)
        plt.savefig(
            # 文件路径+文件名
            os.path.join(self.path_save, f'curve_{self.file_name}.png'),
            # @dpi 每英寸像素数，可以理解为清晰度或细腻度
            dpi = 1000
        )
        plt.close()
    def plt_multicurve(
            self,
            colors,
            legend_text=None,
            legend_size=None,
            x_range=None,
            y_range=None,
            invert_yaxis=False,
    ):
        if self.labels is None:
            for i in range(len(self.y)):
                plt.plot(
                    self.x[i],
                    self.y[i],
                    # self.styles[i],
                    color=colors[i],
                    markersize = 6.5,
                    linewidth = 2,
                )
        else:
            for i in range(len(self.y)):
                plt.plot(
                    self.x[i],
                    self.y[i],
                    # self.styles[i],
                    color=colors[i],
                    markersize = 6.5,
                    linewidth = 2,
                    label = self.labels[i]
                )
        if legend_text is not None:
            plt.legend(
                labels=legend_text,
                loc='best',
                frameon=False,
                fontsize=legend_size
            )    # 去掉图例边框，设置字体
        plt.xlabel('%s' %self.label_x)
        plt.ylabel('%s' %self.label_y)
        if x_range is None:
            plt.xlim(xmin=min(self.x[0]), xmax=max(self.x[0]))
        else:
            plt.xlim(xmin=x_range[0], xmax=x_range[1])
        if y_range is not None:
            plt.ylim(ymin=y_range[0], ymax=y_range[1])
        plt.grid(linestyle = '--')  # 添加网格线
        if invert_yaxis:
            plt.gca().invert_yaxis()
        plt.savefig(
            # 文件路径+文件名
            os.path.join(self.path_save, '%s.png' %self.file_name),
            # @dpi 每英寸像素数，可以理解为清晰度或细腻度
            dpi = 600
        )
        plt.close()

    def plt_curve_with_points(
            self,
            colors,
            scatter_x,
            scatter_y,
            markers=None,
            legend_text=None,
            legend_size=None,
            x_range=None,
            y_range=None,
            invert_yaxis=False,
            markersize=8,
            linewidth=3,
    ):
        """
        绘制曲线 + 点
        markers: 点样式列表，比如 ['o','s','^','d']，若 None 则自动循环
        """

        # 默认点样式循环
        default_markers = ['o', 's', '^', 'd', 'v', 'p', '*', 'x']
        default_colors = ['k', 'g', 'r', 'c', 'm', 'y', 'b']
        if markers is None:
            markers = [default_markers[i % len(default_markers)] for i in range(len(self.y))]

        for i in range(len(self.y)):
            print(self.y[i])
            # 绘制曲线
            plt.plot(
                self.x[i],
                self.y[i],
                color=colors[i],
                linewidth=linewidth
            )
        for i in range(len(scatter_y)):
            # 绘制点
            plt.scatter(
                scatter_x[i],
                scatter_y[i],
                color=default_colors[i],
                marker=markers[i],
                s=markersize ** 2,   # s 为面积
            )

        if legend_text is not None:
            plt.legend(
                labels=legend_text,
                loc='best',
                frameon=False,
                fontsize=legend_size
            )

        plt.xlabel('%s' % self.label_x)
        plt.ylabel('%s' % self.label_y)

        # X 轴范围
        if x_range is None:
            plt.xlim(xmin=min(self.x[0]), xmax=max(self.x[0]))
        else:
            plt.xlim(xmin=x_range[0], xmax=x_range[1])

        # Y 轴范围
        if y_range is not None:
            plt.ylim(ymin=y_range[0], ymax=y_range[1])

        plt.grid(linestyle='--')

        # Y 轴反转
        if invert_yaxis:
            plt.gca().invert_yaxis()

        # 保存图片
        plt.savefig(
            os.path.join(self.path_save, '%s.png' % self.file_name),
            dpi=600
        )
        plt.close()


    def plt_bar(self, text, text_size, text_position):
        '''
        绘制柱状图
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
        # 设置文本位置为右上角
        text_x = xlim[1] - text_position[0] * (xlim[1] - xlim[0])
        text_y = ylim[1] - text_position[1] * (ylim[1] - ylim[0])
        # 添加文本
        if text is not None:
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

class Image3D:
    def __init__(self, fontsize):
        fig = plt.figure()
        self.ax = fig.add_subplot(111, projection='3d')
        self.fontsize = fontsize
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
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        # 设置轴的等间距显示
        self.ax.set_box_aspect([1, 1, 1])  # x, y, z 轴的比例相同
        plt.show()

    def scatters_and_surface(
            self,
            scatters_points,
            grid_x,
            grid_y,
            grid_z,
            text_position,
            text
    ):
        scatters_points_x = scatters_points[:, 0]
        scatters_points_y = scatters_points[:, 1]
        scatters_points_z = scatters_points[:, 2]
        # 绘制三维曲面
        self.ax.plot_surface(
            grid_x,
            grid_y,
            grid_z,
            cmap='viridis',
            alpha=0.9,
            edgecolor='none'    # 定义网格线颜色，'none' 表示没有网格线
        )
        # 绘制点
        self.ax.scatter(
            scatters_points_x,
            scatters_points_y,
            scatters_points_z,
            color='red',
            label='Points'
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
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        plt.show()

class FileUtils:
    def __init__(self, path):
        self.path = path

    # search: 正则表达式
    def remove_search(self, search_list):
        files = []
        for i in range(len(search_list)):
            file = glob.glob(os.path.join(self.path, search_list[i]))
            files += file
        for file in files:
            os.remove(file)

    def remove(self, file_name):
        '''
        删除文件
        :return:
        '''
        file = os.path.join(self.path, file_name)
        try:
            os.remove(file)
        except:
            print(f'File {file} deleted')

    def create_folder(self, folder_name=None):
        '''
        创建文件夹
        :param path:        文件夹路径
        :param folder_name: 文件夹名
        :return:
        '''
        if folder_name is not None:
            folder = os.path.join(self.path, folder_name)
        else:
            folder = self.path
        try:
            os.makedirs(folder)
        except:
            print(f'[INFO] Folder already exits: {folder}')

    def get_file_names_by_type(self, search_string):
        # 获取指定路径下所有指定后缀文件名称
        all_files = os.listdir(self.path)
        bmp_files = [os.path.splitext(file)[0] for file in all_files if file.endswith(f'{search_string}')]
        return bmp_files

    def filter_filenames(self, search_string, suffix_to_remove):
        '''
        在指定路径下检索包含特定字符串的文件名，并移除指定后缀
        参数:
        path -- 要搜索的目录路径
        search_string -- 文件名中需要包含的字符串
        suffix_to_remove -- 需要从文件名中移除的后缀
        返回:
        处理后的文件名列表
        '''
        filtered_files = []
        # 遍历指定路径下的所有文件
        for filename in os.listdir(self.path):
            # 检查是否为文件且包含搜索字符串
            if os.path.isfile(os.path.join(self.path, filename)) and search_string in filename:
                # 移除指定后缀（如果文件名以该后缀结尾）
                if suffix_to_remove and filename.endswith(suffix_to_remove):
                    processed_name = filename[:-len(suffix_to_remove)]
                else:
                    processed_name = filename
                filtered_files.append(processed_name)
        return filtered_files

    def filter_star_filenames(self, search_string, suffix_to_remove):
        """
        在指定路径下检索文件名开头包含特定字符串的文件，并移除指定后缀
        参数:
        path -- 要搜索的目录路径
        search_string -- 文件名开头需要包含的字符串
        suffix_to_remove -- 需要从文件名中移除的后缀
        返回:
        处理后的文件名列表
        """
        filtered_files = []
        # 遍历指定路径下的所有文件
        for filename in os.listdir(self.path):
            # 检查是否为文件且文件名以搜索字符串开头
            full_path = os.path.join(self.path, filename)
            if os.path.isfile(full_path) and filename.startswith(search_string):
                # 移除指定后缀（如果文件名以该后缀结尾）
                if suffix_to_remove and filename.endswith(suffix_to_remove):
                    processed_name = filename[:-len(suffix_to_remove)]
                else:
                    processed_name = filename
                filtered_files.append(processed_name)
        return filtered_files

    def find_dirs_with_keyword(self, keyword):
        """
        返回指定路径下所有包含 keyword 的文件夹名（不递归）
        """
        print(f'[INFO] 路径：{self.path}')
        return [
            name for name in os.listdir(self.path)
            if os.path.isdir(os.path.join(self.path, name)) and keyword in name
        ]

    def copy(self, name, dst_dir, new_name=None):
        """
        复制文件到指定路径并重命名
        参数:
            src_file: 源文件路径（包含文件名）
            dst_dir: 目标文件夹路径
            new_name: 新文件名（可包含扩展名）
        """
        src_file = os.path.join(self.path, name)
        print(src_file)
        # 如果目标目录不存在则创建
        os.makedirs(dst_dir, exist_ok=True)
        if new_name is None:
            shutil.copy(src_file, dst_dir)
        else:
            # 目标完整路径
            dst_file = os.path.join(dst_dir, new_name)
            # 执行复制并重命名
            shutil.copy2(src_file, dst_file)
            print(f"文件已复制并重命名为: {dst_file}")

    def delete_files_with_string(self, search_string):
        """
        删除指定目录及其子目录中所有文件名包含特定字符串的文件

        参数:
            directory (str): 要搜索的根目录路径
            search_string (str): 文件名中要匹配的字符串（默认为'ssss'）
        """
        deleted_count = 0
        # 遍历目录树
        for root, dirs, files in os.walk(self.path):
            for file in files:
                # 检查文件名是否包含目标字符串
                if search_string in file:
                    file_path = os.path.join(root, file)
                    try:
                        os.remove(file_path)
                        print(f"已删除: {file_path}")
                        deleted_count += 1
                    except Exception as e:
                        print(f"删除失败 [{file_path}]: {str(e)}")
        print(f"\n操作完成。共删除 {deleted_count} 个文件。")

    def modify_multiple_lines(self, modifications, backup=False):
        '''
        修改多个指定行，path为文件路径+文件名
        :param modifications: 字典，格式为 {行号: 新内容}
        :param backup: 是否创建备份
        :return:
        '''
        # 创建备份
        if backup:
            import shutil
            shutil.copy2(self.path, f'{self.path}.bak')
            print(f"已创建备份: {self.path}.bak")
        with open(self.path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        # 应用修改
        for line_num, new_content in modifications.items():
            idx = line_num - 1
            if 0 <= idx < len(lines):
                if new_content is None:
                    lines[idx] = ''  # 删除行
                else:
                    lines[idx] = (new_content + '\n'
                                  if not new_content.endswith('\n')
                                  else new_content)
            else:
                print(f"警告: 行号 {line_num} 超出文件范围")
        # 写入文件
        with open(self.path, 'w', encoding='utf-8') as f:
            f.writelines(lines)
        print(f"已修改文件 {self.path}")

class CSVUtils:
    def __init__(self, path, file_name):
        '''
        csv数据
        :param path:        文件路径
        :param file_name:   文件名
        '''
        self.path = path
        self.file_name = file_name
        self.file = os.path.join(self.path, f'{self.file_name}.csv')

    def read(self):
        '''
        以pandas形式获取csv数据
        :return:
        '''
        df = pd.read_csv(self.file)
        return df

    def read_data(self):
        '''
        以列表形式获取csv每行数据
        :return:
        '''
        data = []
        # 打开并读取 CSVUtils 文件
        with open(self.file, mode='r') as file:
            reader = csv.reader(file)
            # 遍历并打印每一行
            for row in reader:
                data.append(row)

    def write_row_data(self, row_data):
        '''
        将一行数据追加到csv中
        :param row_data:
        :return:
        '''
        file_csv = os.path.join(self.path, f'{self.file_name}.csv')
        # 打开或创建一个CSV文件，并将一行数据写入其中
        with open(file_csv, mode='a', newline='') as file:
            writer = csv.writer(file)
            # 写入一行数据
            writer.writerow(row_data)

    def remove(self):
        '''
        删除文件
        :return:
        '''
        try:
            os.remove(self.file)
        except:
            print(f'File {self.file} deleted')