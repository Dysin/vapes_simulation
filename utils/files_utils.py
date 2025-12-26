'''
图片、视频库
Author: Dysin
Time:   2024.06.16
'''

import os
import csv
import glob
import shutil
import pandas as pd

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