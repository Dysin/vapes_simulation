'''
@Desc:   
@Author: Dysin
@Date:   2024/8/4
'''

import subprocess

def run_ansa(path_ansa, file_script, bool_gui):
    '''
    运行ANSA
    :param path_ansa:   ANSA安装路径
    :param file_script: python文件
    :param bool_gui:  是否采用GUI
    :return:
    '''
    command = '%s\\ansa64.bat -exec load_script:\'%s\' -nolauncher' % (path_ansa, file_script)
    print(command)
    if bool_gui is False:
        command += ' -nogui'
    subprocess.run(['cmd', '/c', command], check=True)

if __name__ == '__main__':
    path_ansa = r'C:\Users\HG\AppData\Local\Apps\BETA_CAE_Systems\ansa_v19.1.1'
    # file_script = r'E:\1_Work\templates\vapes_simulation\source\analysis\mesh_ansa\morph.py'

    file_script = r'E:\1_Work\templates\vapes_simulation\source\analysis\mesh_ansa\output_mesh.py'
    run_ansa(path_ansa, file_script, bool_gui=False)