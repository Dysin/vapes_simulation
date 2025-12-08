'''
@Desc:   
@Author: Dysin
@Date:   2025/12/1
'''

import win32com.client
import time
import os

# SolidWorks 文档类型常量
SW_DOC_PART = 1

# 修改你自己系统中的模板路径（必需）
# SolidWorks 默认模板通常长这样（请按自己版本修改年份）
DEFAULT_PART_TEMPLATE = r"C:\Users\Public\Documents\SOLIDWORKS\SOLIDWORKS 2024\samples\tutorial\advdrawings\part.prtdot"


def create_cylinder(diameter_mm=10, height_mm=20, template=DEFAULT_PART_TEMPLATE):
    sw = win32com.client.DispatchEx("SldWorks.Application")
    sw.Visible = True

    # 1) 新建零件
    print("Creating new part...")
    model = sw.NewDocument(template, 0, 0.0, 0.0)
    if model is None:
        raise RuntimeError("无法创建新零件，请检查模板路径是否正确。")

    # sw.ActivateDoc2("Part1", False, 0)
    doc = sw.ActiveDoc
    print('0000')
    sketchMgr = doc.SketchManager
    print('111')
    sketchMgr.CreateRectangle(0, 0, 0, 0.1, 0.1, 0)
    print('222')
    # 2) 选中 Front Plane
    boolstatus = model.Extension.SelectByID2(
        "Front Plane",  # 名称
        "PLANE",  # 类型
        0, 0, 0,
        False, 0, None, 0
    )
    if not boolstatus:
        raise RuntimeError("无法选中 Front Plane。")

    print("Starting sketch on Front Plane...")

    # 3) 插入草图
    sketchMgr = model.SketchManager
    sketchMgr.InsertSketch(True)

    # 设定圆的半径（SolidWorks 的 CreateCircleByRadius 用半径）
    radius = diameter_mm / 2.0 / 1000.0  # 转成米（SW 内部单位是米）

    # 4) 在原点画圆
    # 这里必须指定三维坐标 (x, y, z)，以米为单位
    circle = sketchMgr.CreateCircleByRadius(0, 0, 0, radius)
    if circle is None:
        raise RuntimeError("创建圆失败。")

    print("Circle created.")

    # 5) 退出草图
    sketchMgr.InsertSketch(False)

    # 6) 选择草图并拉伸
    # 草图名一般是 "Sketch1"
    boolstatus = model.Extension.SelectByID2(
        "Sketch1",
        "SKETCH",
        0, 0, 0,
        False, 0, None, 0
    )
    if not boolstatus:
        raise RuntimeError("无法选中 Sketch1。")

    print("Extruding...")

    featMgr = model.FeatureManager

    # FeatureExtrusion2 的参数非常长，所以这里使用简单模式：
    # Blind 拉伸，方向 1，高度为 height_mm
    height_m = height_mm / 1000.0  # mm → m

    feature = featMgr.FeatureExtrusion2(
        True,  # 草图是否合并实体
        False,  # 是否反向
        False,  # 是否向两边
        0,  # blind（拉伸类型）
        0,  # second direction type
        height_m,  # 深度（米）
        0.0,
        False, False, False, False,
        0.0, 0.0,
        False, False, False, False,
        True,  # 是否自动选择轮廓
        True  # propagate feature
    )
    if feature is None:
        raise RuntimeError("拉伸失败，请检查参数。")

    print("Cylinder created!")
    return model


if __name__ == "__main__":
    create_cylinder(
        diameter_mm=12,
        height_mm=40
    )
