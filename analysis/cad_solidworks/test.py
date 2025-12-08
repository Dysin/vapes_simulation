import FreeCAD
import Part  # FreeCAD 的几何模块
import Import

# 设置 FreeCAD 文档
doc = FreeCAD.newDocument("ImportSTEP")

# STEP 文件路径
step_file = r"E:\1_Work\your_model.stp"  # 注意修改为你自己的路径

# 导入 STEP 文件
shape = Import.open(step_file)

# 刷新文档视图（如果你在 GUI 下使用）
doc.recompute()

print("STEP 文件导入完成")
