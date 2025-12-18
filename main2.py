'''
@Desc:   主函数入口
@Author: Dysin
@Date:   2025/11/20
'''

import numpy as np
from workflow.workflow_airway import WorkflowRANS

if __name__ == '__main__':
    vape_name = 'VP158-A'
    ver_num = '20251218'
    mesh_user_name = None

    vape_type = '一次性'
    workflow = WorkflowRANS(vape_name, ver_num, mesh_folder='origin')
    workflow.spf_default_flow_rates(vape_type)

    input_params_range = [
        [-0.6, -0.1],
        [-0.1, 0.1],
        [0.2, 0.5],
        [-1.0, -0.5]
    ]  # 2025.12.09 pm

    # workflow.spf_uq(
    #     bool_doe=True,
    #     bool_morph=True,
    #     bool_sim=True,
    #     bool_latex=False,
    #     input_params_range=input_params_range,
    #     batch_id=5, # 从1开始
    #     sample_num=10,
    #     flow_rate=22.5,
    #     sim_star_num=1 # 从1开始
    # )