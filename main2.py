'''
@Desc:   主函数入口
@Author: Dysin
@Date:   2025/11/20
'''

import numpy as np
from workflow.workflow_airway import WorkflowRANS

if __name__ == '__main__':
    vape_name = 'F01-RD03'
    ver_num = '20251126'
    mesh_user_name = None
    vape_type = '换弹'
    # workflow = WorkflowRANS(vape_name, ver_num, mesh_folder='optimization')
    workflow = WorkflowRANS(vape_name, ver_num, mesh_folder='origin')
    workflow.rans_spf_default_flow_rates(vape_type, mesh_user_name)

    basic_flow_rates = [17.5, 20.5, 22.5]
    flow_rates = np.arange(25, 45, 2.5).tolist()
    flow_rates += basic_flow_rates
    flow_rates = sorted(set(flow_rates))
    print(flow_rates)
    # workflow.rans_spf_user_flow_rates(
    #     vape_type=vape_type,
    #     flow_rates=flow_rates,
    #     mesh_user_name=mesh_user_name,
    #     star_num=0
    # )
    workflow.rans_spf_experiment_compare(
        flow_rates=flow_rates,
        mesh_user_name=mesh_user_name
    )