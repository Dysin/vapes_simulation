'''
@Desc:   主函数入口
@Author: Dysin
@Date:   2025/11/20
'''

import numpy as np
from workflow.workflow_airway import WorkflowRANS

if __name__ == '__main__':
    vape_name = 'VP353'
    ver_num = '20251201'
    # mesh_user_name = 'modify_r0p4'

    vape_type = '一次性'
    workflow = WorkflowRANS(vape_name, ver_num, mesh_folder='doe')

    input_params_range = [
        [-0.6, -0.1],
        [-0.1, 0.1],
        [0.2, 0.5],
        [-1.0, -0.5]
    ]  # 2025.12.09 pm

    workflow.spf_uq(
        bool_doe=False,
        bool_morph=False,
        bool_sim=False,
        bool_latex=False,
        input_params_range=input_params_range,
        batch_id=4,
        sample_num=10,
        flow_rate=22.5,
        sim_star_num=5
    )