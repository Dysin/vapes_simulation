'''
@Desc:   Solidworks 工具类
@Author: Dysin
@Date:   2025/12/1
'''

import os
import win32com.client

class SolidworksUtils(object):
    def __init__(self, path_root):
        self.part_template = r'C:\Users\Public\Documents\SOLIDWORKS\SOLIDWORKS 2024\samples\tutorial\advdrawings\part.prtdot'
        self.sw = self.create_instance()
        self.path_root = path_root

    def test(self):
        sw_model = self.sw.ActiveDoc
        print(sw_model)
        if sw_model is None:
            print("未打开 SolidWorks 文档")
        else:
            if sw_model.GetType == 1:  # swDocPART = 1
                part = sw_model
                bodies = part.GetBodies2(2, False)
                if bodies:
                    for i, b in enumerate(bodies):
                        name = b.Name
                        if not name:
                            name = f"Body_{i}"
                        print(name)
                else:
                    print("零件中没有 SolidBody")
            elif sw_model.GetType == 2:  # swDocASSEMBLY = 2
                asm = sw_model
                comps = asm.GetComponents(True)  # True = 递归
                for comp in comps:
                    comp_model = comp.GetModelDoc2()
                    if comp_model is None:
                        continue
                    bodies = comp_model.GetBodies2(2, False)
                    if bodies:
                        for i, b in enumerate(bodies):
                            name = b.Name
                            if not name:
                                name = f"Body_{i}"
                            print(f"Component: {comp.Name2}, Body: {name}")
                    else:
                        print(f"Component: {comp.Name2} 没有 SolidBody")

    def create_instance(self):
        # 启动 SolidWorks
        # sw = win32com.client.Dispatch("SldWorks.Application") # 启动了SW后使用
        sw = win32com.client.DispatchEx("SldWorks.Application") # 未启动SW使用
        sw.Visible = True
        print("SolidWorks 启动为英文界面模式")
        return sw

    def input_stp(self, file_name):
        file_stp = os.path.join(self.path_root, f'{file_name}.stp')
        print(file_stp)
        sw_model = self.sw.OpenDoc(file_stp, 1)

    def create_new_part(self):
        print("Creating new part...")
        model = self.sw.NewDocument(self.part_template, 0, 0.0, 0.0)
        if model is None:
            raise RuntimeError("无法创建新零件，请检查模板路径是否正确。")
        return model

    def generate_print_bodies_macro(self, output_path):
        # 用一行一行写的方式生成宏
        lines = [
            "Option Explicit",
            "Sub main()",
            "    Dim swApp As SldWorks.SldWorks",
            "    Set swApp = Application.SldWorks",
            "",
            "    Dim swModel As ModelDoc2",
            "    Set swModel = swApp.ActiveDoc",
            "",
            "    If swModel Is Nothing Then",
            "        MsgBox \"请打开零件或装配模型！\"",
            "        Exit Sub",
            "    End If",
            "",
            "    Dim swPart As Object",
            "    Set swPart = swModel",
            "",
            "    Dim vBodies As Variant",
            "    vBodies = swPart.GetBodies2(swSolidBody, False)",
            "",
            "    If IsEmpty(vBodies) Then",
            "        MsgBox \"没有找到 SolidBody\"",
            "        Exit Sub",
            "    End If",
            "",
            "    Dim swBody As Object",
            "    Dim i As Long",
            "    For i = 0 To UBound(vBodies)",
            "        Set swBody = vBodies(i)",
            "        Dim sName As String",
            "        sName = swBody.Name",
            "        If sName = \"\" Then",
            "            sName = \"Body_\" & i",
            "        End If",
            "        Debug.Print \"Body: \" & sName",
            "    Next i",
            "",
            "    MsgBox \"SolidBody 打印完成！请查看 Immediate Window (Ctrl+G)\"",
            "End Sub"
        ]

        with open(output_path, 'w', encoding='utf-8') as f:
            for line in lines:
                f.write(line + "\n")

        print(f"宏已生成: {output_path}")
        return output_path

    def run_macro(self, file_macro):
        self.sw.RunMacro(file_macro, "", "")

if __name__ == '__main__':
    path = r'E:\1_Work\active\airway_analysis\VP353\geometry'
    stp_name = 'orig_000-vp353-unit_asm'
    macro = r'E:\1_Work\active\airway_analysis\VP353\geometry\test.swp'
    sw_utils = SolidworksUtils(path_root=path)
    sw_utils.create_new_part()
    # sw_utils.generate_print_bodies_macro(macro)
    # sw_utils.run_macro(macro)
    sw_utils.input_stp(stp_name)