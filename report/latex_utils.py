'''
@Desc:   输出latex报告
@Author: Dysin
@Date:   2025/10/13
'''

import sys
import os.path
import re
import subprocess
from utils.files_utils import FileUtils

class LatexUtils:
    def __init__(
            self,
            vape_name,
            version_number,
            path,
            date=None
    ):
        self.vape_name = vape_name
        self.ver_num = version_number
        self.date = date if date else r"\today"  # 默认为今天的日期
        self.content = ""  # 存放主体内容
        self.packages = []  # 存放需要使用的宏包
        self.path = path

    # def image_name_to_list(self, image_name):
    #     # 匹配 pattern 如：atomization_velocity_y0.0394
    #     match = re.match(r'([a-zA-Z_]+)_([xyz])([\-+]?\d+\.?\d*)$', image_name)
    #     if match:
    #         var_name = match.group(1)  # "atomization_velocity"
    #         axis = match.group(2)  # "x", "y" 或 "z"
    #         value = match.group(3)  # "0.0394" 等
    #         value = float(value)  # 转换为浮点数
    #         return [var_name.split('_')[-1], axis, value]  # 返回最后部分的变量名, 坐标轴和数值
    #
    #     return image_name  # 如果格式不匹配，返回原字符串

    def image_name_to_list(self, image_name):
        # 匹配：变量_轴值  例如 atomization_velocity_z5e-05
        pattern = r'([a-zA-Z_]+)_([xyz])([\-+]?\d*\.?\d+(?:[eE][\-+]?\d+)?)$'
        match = re.match(pattern, image_name)
        if match:
            var_name = match.group(1)  # 变量名，如 atomization_velocity
            axis = match.group(2)  # x/y/z
            value_str = match.group(3)  # 数值部分，含科学计数法
            try:
                value = float(value_str)
            except ValueError:
                raise ValueError(f"无法解析数值：{value_str}")
            # 返回最后一个字段的名字（如 velocity）
            return [var_name.split('_')[-1], axis, value]
        # 不匹配就原样返回，保证程序安全
        return image_name

    def pair_caption_list(
            self,
            image_names,
            unit="mm",
            left_tag="（左）",
            right_tag="（右）",
            suffix="云图"
    ):
        """
        将类似 [['pressure','y',0.0], ['pressure','y',2.0], ...] 的列表
        转换为每两个为一组的描述性标题列表
        参数：
            data_list: list[list]  输入三元组列表，例如 [['pressure','y',0.0], ...]
            unit: str              数值单位（默认 'mm'）
            left_tag: str          左图标签（默认 '(左)'）
            right_tag: str         右图标签（默认 '(右)'）
            suffix: str            后缀描述（默认 '云图'）

        返回：
            list[str]  格式化后的字符串列表
        """
        data_list = []
        for image_name in image_names:
            data_list.append(self.image_name_to_list(image_name))
        result = []
        for i in range(0, len(data_list), 2):
            if i + 1 < len(data_list):
                var1, direction1, val1 = data_list[i]
                var2, direction2, val2 = data_list[i + 1]
                # 方向相同则使用一次
                direction = direction1
                # 生成字符串
                if unit == 'mm':
                    val1 = f'{val1 * 1000:.2f}'
                    val2 = f'{val2 * 1000:.2f}'
                elif unit == 'cm':
                    val1 = f'{val1 * 100:.2f}'
                    val2 = f'{val2 * 100:.2f}'
                else:
                    val1 = '{:.5f}'.format(val1)
                    val2 = '{:.5f}'.format(val2)
                caption = f"{direction}={val1}{unit}{left_tag}、{direction}={val2}{unit}{right_tag} {var1}{suffix}"
                result.append(caption)
        return result

    def preamble(self, header=None):
        image_path = os.path.join(self.path, 'images')
        image_path = image_path.replace('\\', '/')
        code = (
            '\\documentclass[12pt]{article}\n'
            '\\usepackage[utf8]{inputenc}\n'
            '\\usepackage{geometry}\n'
            '\\usepackage{ctex}  % 支持中文\n'
            '\\usepackage{graphicx} % 用于插入图片\n'
            '\\usepackage{caption} % 自定义图注\n'
            '\\usepackage{float} % 在导言区添加\n'
            '\\usepackage{subcaption} % 用于子图\n'
            '\\usepackage{fancyhdr} % 引入页眉页脚宏包\n'
            '\\usepackage{booktabs} % 用于绘制三线表\n'
            '\\usepackage{amsmath}  % 在导言区添加\n'
            '\\usepackage{hyperref} % 引用标签\n'
            '\\geometry{a4paper, margin=1in}\n\n'
        )
        if header is not None:
            code += (
                '\\pagestyle{fancy} %设置页面样式为fancy\n'
                '\\fancyhf{} %清除所有页眉页脚设置\n'
                f'\\fancyhead[L]{{{header}}} %中页眉\n'
                '\\renewcommand{\\headrulewidth}{0.2pt} %页眉下方横线宽度\n\n'
            )
        code += '\\begin{document}\n\n'
        code += (
            '% 图片路径设置（根据实际情况修改）\n'
            f'\\graphicspath{{{{{image_path}/}}}} % 图片存放在images文件夹中\n\n'
        )
        return code

    def insert_figure(
            self,
            name,
            title,
            tag,
            image_height=3
    ):
        code = (
            '\\begin{figure}[H]\n'
            '\t\\centering\n'
            f'\t\\includegraphics[height={image_height}cm]{{{name}.png}} %图片文件名\n'
            f'\t\\caption{{{title}}}\n'
            f'\t\\label{{fig: {tag}}}\n'
            '\\end{figure}\n\n'
        )
        return code

    def insert_subfigures(
            self,
            rows,
            cols,
            image_names,
            captions,
            main_caption,
            max_images_per_page=6
    ):
        """
        生成动态的 LaTeX 图片排版代码，自动换页
        参数:
        - rows: 图片的行数
        - cols: 图片的列数
        - image_names: 图片文件名列表，按行列顺序排列
        - captions: 每张图片的标题列表，按行列顺序排列
        - main_caption: 总标题
        - max_images_per_page: 每页最多显示的图片数，默认6
        返回:
        - 生成的 LaTeX 图片插入代码
        """
        latex_code = f"\\begin{{figure}}[H]\n    \\centering\n    \\caption{{{main_caption}}}  % 总标题\n"
        # 图片的总数量
        total_images = rows * cols
        image_index = 0
        for i in range(rows):
            latex_code += "    % 第{}行图片\n".format(i + 1)
            for j in range(cols):
                img_file = image_names[image_index]
                caption = captions[image_index]
                latex_code += f"    \\begin{{subfigure}}[b]{{0.45\\textwidth}}\n"
                latex_code += f"        \\includegraphics[width=\\textwidth]{{{img_file}}}\n"
                latex_code += f"        \\caption{{{caption}}}\n"
                latex_code += f"        \\label{{fig:sub{image_index + 1}}}\n"
                latex_code += "    \\end{subfigure}\n"
                # 为了行间的间隔，如果不是最后一列，插入 \hfill
                if j < cols - 1:
                    latex_code += "    \\hfill\n"
                image_index += 1
                # 如果当前图片数量达到最大限制，则插入换页命令
                if image_index % max_images_per_page == 0 and image_index < total_images:
                    latex_code += "\\clearpage\n"  # 使用 \clearpage 强制换页
            latex_code += "\n"
        latex_code += "    \\label{fig:cloud_images}\n\\end{figure}\n"
        return latex_code

    def insert_figures_two_per_row(self, image_names, captions, main_captions):
        """
        生成多页 LaTeX 图片代码（每行两列）

        参数：
            image_names: list[str]      图片文件名列表（按顺序）
            captions: list[str] | None  子图标题（可选）
            main_captions: list[str]    每页主标题（与页数对应）
        """
        latex_all = ""
        num_images = len(image_names)
        # 确保图片数量是偶数，不足则补空白
        if num_images % 2 != 0:
            image_names.append("")
            if captions is not None:
                captions.append("")
        num_figures = (num_images + 1) // 2  # 每页两张图片
        print(f'[INFO] num figures: {num_figures}')
        print(f'[INFO] main captions num: {len(main_captions)}')
        print(f'[INFO] main captions: {main_captions}')
        assert len(main_captions) >= num_figures, "main_captions 数量不足"
        for i in range(0, num_images, 2):
            page_index = i // 2
            group = image_names[i:i + 2]
            subcaps = captions[i:i + 2] if captions is not None else [None, None]
            main_caption = main_captions[page_index]
            # 每个 figure 块
            latex_code = "\\begin{figure}[H]\n    \\centering\n"
            for j, img in enumerate(group):
                if not img:
                    continue
                latex_code += f"    \\begin{{subfigure}}[b]{{0.4\\textwidth}}\n"
                latex_code += f"        \\includegraphics[width=\\textwidth]{{{img}}}\n"
                if subcaps[j]:  # 如果 captions 存在
                    latex_code += f"        \\caption{{{subcaps[j]}}}\n"
                latex_code += f"        \\label{{fig:{img}}}\n"
                latex_code += f"    \\end{{subfigure}}\n"
                if j == 0:
                    latex_code += "    \\hfill\n"
            # 主标题放在子图之后
            latex_code += f"\n    \\caption{{{main_caption}}}\n"
            latex_code += f"    \\label{{fig:page_{page_index + 1}}}\n\\end{{figure}}\n\n"
            latex_all += latex_code
        return latex_all

    def insert_figures_per_row(self, image_names, captions, main_captions):
        """
        生成多页 LaTeX 图片代码（每行 m 列，共 n 行）
        参数：
            image_names: list[list[str]]   二维列表，每行表示一行图片（n 行 m 列）
            captions: list[list[str]] | None  与 image_names 对应的子图标题（可选）
            main_captions: list[str]        每页主标题（与 image_names 行数对应）
        """
        latex_all = ""
        n_rows = len(image_names)
        print(f"[INFO] total rows: {n_rows}")
        print(f"[INFO] main captions num: {len(main_captions)}")
        assert len(main_captions) >= n_rows, "main_captions 数量不足"
        for i, row_images in enumerate(image_names):
            m_cols = len(row_images)
            row_captions = captions[i] if captions is not None else [None] * m_cols
            main_caption = main_captions[i]
            # 每个 figure 块
            latex_code = "\\begin{figure}[H]\n    \\centering\n"
            lab_imgs = []
            for j, img in enumerate(row_images):
                if not img:
                    continue
                width = round(0.9 / m_cols, 2)  # 控制每张图的宽度
                latex_code += f"    \\begin{{subfigure}}[b]{{{width}\\textwidth}}\n"
                latex_code += f"        \\includegraphics[width=\\textwidth]{{{img}}}\n"
                if row_captions[j]:
                    latex_code += f"        \\caption{{{row_captions[j]}}}\n"
                latex_code += f"        \\label{{fig:{img}}}\n"
                latex_code += f"    \\end{{subfigure}}\n"
                if j != m_cols - 1:
                    latex_code += "    \\hfill\n"
                lab_imgs.append(img)
            lab_main = f'{lab_imgs[0]}_{lab_imgs[1]}'
            # 主标题
            latex_code += f"\n    \\caption{{{main_caption}}}\n"
            latex_code += f"    \\label{{fig:{lab_main}}}\n\\end{{figure}}\n\n"
            latex_all += latex_code
        return latex_all

    def insert_table(
            self,
            data,
            caption=None,
            label=None,
            alignment=None,
            position="htbp"
    ):
        """
        生成 LaTeX 三线表代码
        参数:
        data -- 二维列表，包含表格数据（第一行通常为表头）
        caption -- 表格标题（可选）
        label -- 表格引用标签（可选）
        alignment -- 列对齐方式字符串，如 "lcr"（可选，默认为所有列居中）
        position -- 表格位置参数，如 "htbp"（可选）
        返回:
        LaTeX 表格代码字符串
        """
        # 确定列数和行数
        n_cols = len(data[0])
        n_rows = len(data)
        # 设置默认对齐方式（所有列居中）
        if alignment is None:
            alignment = "c" * n_cols
        elif len(alignment) != n_cols:
            alignment = "c" * n_cols  # 如果提供的对齐方式不匹配列数，使用默认值
        # 开始构建表格代码
        code = []
        # 添加表格浮动环境
        code.append(r"\begin{table}[" + position + "]")
        code.append(r"\centering")
        # 添加标题和标签
        if caption:
            code.append(r"\caption{" + caption + "}")
        if label:
            code.append(r"\label{" + label + "}")
        # 开始表格环境
        code.append(r"\begin{tabular}{" + alignment + "}")
        code.append(r"\toprule")  # 顶部线
        # 添加表头（第一行）
        header = " & ".join(str(cell) for cell in data[0]) + r" \\"
        code.append(header)
        code.append(r"\midrule")  # 中间线
        # 添加数据行
        for row in data[1:]:
            row_str = " & ".join(str(cell) for cell in row) + r" \\"
            code.append(row_str)
        # 结束表格
        code.append(r"\bottomrule")  # 底部线
        code.append(r"\end{tabular}")
        code.append("\\end{table}\n\n")
        # 添加必要的包引用
        code.insert(0, r"% 请确保在文档开头添加以下包：")
        code.insert(1, r"% \usepackage{booktabs}")
        return "\n".join(code)

    def first_page(self, main_title='结构及气道仿真分析'):
        code = (
            '\\begin{center}\n'
            '\t\\vspace*{3cm} % 顶部垂直间距调整\n\n'
            '\t\\begin{figure}[H]\n'
            '\t\\centering\n'
            '\t\\includegraphics[height=5cm]{E:/1_Work/images/logo.jpg}\n'
            '\t\\end{figure}\n\n'
            '\t\\vspace{3\\baselineskip}\n\n'
            '\t% 主标题（加大字号）\n'
            f'\t{{\\Huge \\textbf{{{main_title}}}}}\n\n'
            '\t\\bigskip\n\n'
            '\t% 副标题\n'
            f'\t{{\\Large {self.vape_name}}}\n\n'    
            '\t% 间隔N行\n'
            '\t\\vspace{10\\baselineskip}\n\n'   
            '\t% 部门信息\n'
            '\t研发工程部 \n'
            '\t\\vspace{1\\baselineskip} % 部门与姓名间距\n\n'
            '\t% 姓名\n'
            '\t丘德新\n\n'
            '\t\\vspace{1\\baselineskip} % 姓名与日期间距\n\n'
            '\t% 日期（自动生成当前日期）\n'
            '\t\\today\n\n'
            '\\end{center}\n\n'
            '\\thispagestyle{empty} % 隐藏第一页页码\n\n'
            '\\newpage\n\n'
            '\\setcounter{page}{1} % 重置页码为1\n\n'
        )
        return code

    def end(self):
        code = (
            '\n\\end{document}\n'
        )
        return code

    def save_to_file(
            self,
            latex_code,
            file_name=None
    ):
        """
        将生成的 LaTeX 文档保存为 `.tex` 文件。
        :param filename: 文件名，默认是 `report.tex`
        """
        if file_name is None:
            file_name = f'{self.vape_name}_{self.ver_num}_结构及气道仿真报告'
        file_tex = os.path.join(self.path, f'{file_name}.tex')
        with open(file_tex, "w", encoding="utf-8") as f:
            f.write(latex_code)
        print(f"Latex file saved as {file_tex}")

    def compile_pdf(self, output_path=None, pdflatex_path=None):
        """
        使用 pdflatex 编译生成 PDF 文件，并输出到指定路径。

        :param output_path: PDF 输出目录（可选，默认为源文件所在目录）
        :param pdflatex_path: pdflatex 可执行文件的完整路径（可选）
        """
        # 获取源文件路径
        file_tex = os.path.join(self.path, f'{self.vape_name}_结构及气道仿真报告.tex')
        # 确定输出目录
        if output_path:
            # 确保输出目录存在
            os.makedirs(output_path, exist_ok=True)
        else:
            output_path = self.path
        # 确定 pdflatex 可执行文件路径
        if pdflatex_path:
            executable = pdflatex_path
        else:
            # 尝试自动查找常见路径
            if sys.platform.startswith('win'):
                executable = "pdflatex.exe"
            else:
                executable = "/usr/bin/pdflatex"
        try:
            # 构建编译命令
            command = [
                executable,
                "-interaction=nonstopmode",  # 非交互模式，遇到错误不暂停
                "-output-directory", output_path,  # 指定输出目录
                file_tex  # 源文件
            ]
            # 执行编译
            result = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='ignore',
            )
            # 检查编译日志
            if "Error" in result.stdout:
                print("编译过程中出现错误:")
                print(result.stdout)
                return False
            print(f"PDF 编译成功: {os.path.join(output_path, self.vape_name + '.pdf')}")
            return True
        except FileNotFoundError:
            print(f"错误: 找不到 pdflatex 可执行文件: {executable}")
            print("请确保已安装 LaTeX 并正确配置路径")
            return False
        except subprocess.CalledProcessError as e:
            print(f"PDF 编译失败，错误代码: {e.returncode}")
            print("编译输出:")
            print(e.stdout)
            print("错误信息:")
            print(e.stderr)
            return False

if __name__ == "__main__":
    results_path = r'D:\1_Work\active\202510_VP322-B\simulation\steady_rans_flow'
    tex_path = r'D:\1_Work\active\202510_VP322-B\file\result_analysis_flow'
    files = FileUtils(results_path)
    image_names = files.get_file_names_by_type('.bmp')
    print(image_names)
    # 使用该类生成报告
    report = LatexUtils(
        vape_name='Vape',
        path=tex_path
    )
    report.save_to_file(tex_path)

