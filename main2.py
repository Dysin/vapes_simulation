'''
@Desc:   主函数入口
@Author: Dysin
@Date:   2025/11/20
'''

import numpy as np
from workflow.workflow_airway import WorkflowRANS

if __name__ == '__main__':
    vape_name = 'VP353'
    ver_num = '20251219'
    mesh_user_name = 'mod_opt02'

    vape_type = '一次性'
    workflow = WorkflowRANS(
        vape_name,
        ver_num,
        mesh_folder='optimization',
        mesh_user_name=mesh_user_name
    )
    # workflow.spf_default_flow_rates(vape_type)

    workflow.spf_user_flow_rates(
        vape_type,
        [17.5],
        star_num=0,
        bool_sim=True,
        bool_post=True
    )

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