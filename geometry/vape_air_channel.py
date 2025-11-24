'''
@Desc:   
@Author: Dysin
@Date:   2025/9/26
'''
import FreeCAD
import Part
import FreeCADGui

# 创建一个新的文档
doc = FreeCAD.newDocument()

# 气道的参数
radius = 5.0  # 半径
length = 100.0  # 长度

# 创建一个圆柱体作为气道
cylinder = Part.makeCylinder(radius, length)

# 将圆柱体添加到文档中
part_object = doc.addObject("Part::Feature", "VapeAirChannel")
part_object.Shape = cylinder

# 刷新文档
doc.recompute()

# 导出为 STP 文件
stp_filename = "vape_air_channel.stp"
Part.export([part_object], stp_filename)

print(f"STP 文件已保存为 {stp_filename}")
