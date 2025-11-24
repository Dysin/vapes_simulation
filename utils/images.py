'''
@Desc:   图片管理器
@Author: Dysin
@Date:   2025/11/24
'''

import os
from PIL import Image, ImageOps

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

if __name__ == '__main__':
    image = ImageUtils()
    path_bmp = r'D:\1_Work\active\202510_VP322-B\simulation\steady_rans_flow'
    path_png = r'D:\1_Work\active\202510_VP322-B\file\result_analysis_flow'
    image_name = 'ff'
    image.bmp_to_png(path_bmp, path_png, image_name, compress_level=1)
