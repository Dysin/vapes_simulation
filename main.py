'''
@Desc:   主函数入口
@Author: Dysin
@Date:   2025/11/20
'''

import numpy as np
from main_workflow.workflow import Workflow

if __name__ == '__main__':
    vape_name = 'VP325-B'
    ver_num = '20251103'
    mesh_user_name = 'level2'
    workflow = Workflow(vape_name, ver_num)
    # workflow.rans_spf_basic_flow_rates(mesh_user_name)

    basic_flow_rates = [17.5, 20.5, 22.5]
    flow_rates = np.arange(25, 45, 2.5).tolist()
    flow_rates += basic_flow_rates
    flow_rates = sorted(set(flow_rates))
    print(flow_rates)
    workflow.rans_spf_user_flow_rates(
        flow_rates=flow_rates,
        mesh_user_name=mesh_user_name
    )