'''
@Desc:   结果分析
@Author: Dysin
@Date:   2025/10/16
'''

class ResultsAnalysis():
    def total_pressure_loss(self, value):
        '''
        总压损失分析，与吸阻、舒适度直接相关。压降越大吸阻越重
        :param value: 进出口总压降
        :return:
        '''
        txt = f'当前气道入口与出口的总压力损失为{value:.2f}Pa。'
        if value < 150:
            txt += (
                '压力损失较小，吸阻较低，气流较为顺畅。\n\n'
            )
        elif 150 <= value < 400:
            txt += (
                '压力损失适中，吸阻有一定程度的增加，但整体流动仍较为顺畅。可以考虑通过微调气道设计来进一步减小压力损失，优化流动。\n\n'
            )
        elif 400 <= value < 600:
            txt += (
                '压力损失较大，吸阻较为明显，可能导致用户吸烟体验不佳。建议对气道进行优化，减少压力损失，降低吸阻。\n\n'
            )
        else:
            txt += (
                '压力损失过大，吸阻明显，容易导致用户吸烟体验不佳。建议对气道进行大幅度优化，减少压力损失，降低吸阻。\n\n'
            )
        return txt

    def sensor_pressure_loss(self, delta_p_total, delta_p_sensor):
        '''
        咪头与出口总压损失
        :param delta_p_total: 出入口总压差
        :param delta_p_sensor: 咪头与出口总压差
        :return:
        '''
        ratio = delta_p_sensor / delta_p_total * 100
        txt = f'咪头与出口的总压力损失为{delta_p_sensor:.2f}Pa，占吸阻的比例为{ratio:.2f}\\%，'
        if ratio < 25:
            txt += (
                '启动灵敏。\n\n'
            )
        elif 25 <= ratio < 50:
            txt += (
                '启动灵敏度偏低。\n\n'
            )
        elif 50 <= ratio < 75:
            txt += (
                '压力损失较大，启动不灵敏。建议优化咪头与烟嘴的连接处设计。\n\n'
            )
        else:
            txt += (
                '压力损失过大，较难启动。强烈建议对咪头与烟嘴连接处进行大幅度调整，减小压力损失。\n\n'
            )
        return txt

    def total_pressure_losses(self, values, flow_rates):
        '''
        总压损失分析，与吸阻、舒适度直接相关。压降越大吸阻越重
        :param value: 进出口总压降
        :return:
        '''
        txt = ''
        for i in range(len(flow_rates)):
            txt += f'当抽吸条件为{flow_rates[i]:.2f}mL/s时，吸阻为{values[i]:.2f}Pa。\n\n'
        # 总结段落
        avg_loss = sum(values) / len(values)
        txt += f'综合来看，平均压力损失约为 {avg_loss:.2f} Pa。'
        if avg_loss < 200:
            txt += '压力损失较小，吸阻较低，气流顺畅，用户体验较为轻盈。\n'
        elif 200 <= avg_loss < 300:
            txt += '整体气道设计合理，吸阻略有感但较为舒适。\n'
        elif 300 <= avg_loss < 500:
            txt += '整体吸阻略偏重，可通过微调气道结构进一步优化。\n'
        elif 500 <= avg_loss < 700:
            txt += '整体吸阻明显，建议优化气道截面或减少局部收缩区域，以降低压降。\n'
        else:
            txt += '整体吸阻较大，需显著优化气道结构（如扩大截面、优化流线过渡）\n'
        return txt

    def sensor_pressure_losses(self, delta_p_totals, delta_p_sensors, flow_rates):
        '''
        咪头与出口总压损失
        :param delta_p_total: 出入口总压差
        :param delta_p_sensor: 咪头与出口总压差
        :return:
        '''
        avg_loss_total = sum(delta_p_totals) / len(delta_p_totals)
        avg_loss_sensor = sum(delta_p_sensors) / len(delta_p_sensors)
        ratio = avg_loss_sensor / avg_loss_total * 100
        txt = f'咪头与出口的平均总压力损失为{avg_loss_sensor:.2f}Pa，占平均吸阻的比例为{ratio:.2f}\\%，'
        if ratio < 25:
            txt += (
                '启动灵敏。\n\n'
            )
        elif 25 <= ratio < 50:
            txt += (
                '启动灵敏度偏低。\n\n'
            )
        elif 50 <= ratio < 75:
            txt += (
                '压力损失较大，启动不灵敏。建议优化咪头与烟嘴的连接处设计。\n\n'
            )
        else:
            txt += (
                '压力损失过大，较难启动。强烈建议对咪头与烟嘴连接处进行大幅度调整，减小压力损失。\n\n'
            )
        return txt

    def default_flow_rates_pressure(
            self,
            total_pressure_sensors,
            total_pressure_losses,
            sensor_pressure_losses
    ):
        '''
        不同默认抽吸条件下的总压及总压损失分析
        :param total_pressure_sensors: 咪头总压
        :param total_pressure_losses: 吸阻
        :param sensor_pressure_losses: 咪头至吸嘴压降
        :return:
        '''
        flow_rates = [17.5, 20.5, 22.5]
        txt = ''
        for i in range(len(flow_rates)):
            txt += (
                f'当抽吸条件为{flow_rates[i]:.2f}mL/s时，'
                f'吸阻为{total_pressure_losses[i]:.2f}Pa，'
                f'咪头处平均压力为{total_pressure_sensors[i]:.2f}Pa，'
            )
            if total_pressure_sensors[i] > -100:
                txt += '此时咪头（阈值-100Pa）不启动。\n\n'
            else:
                txt += '此时咪头（阈值-100Pa）可启动。\n\n'
        # 总结段落
        avg_loss = sum(total_pressure_losses) / len(flow_rates)
        txt += f'综合来看，平均压力损失约为 {avg_loss:.2f} Pa。'
        if avg_loss < 200:
            txt += '压力损失较小，吸阻较低，气流顺畅，用户体验较为轻盈。\n'
        elif 200 <= avg_loss < 300:
            txt += '整体气道设计合理，吸阻略有感但较为舒适。\n'
        elif 300 <= avg_loss < 500:
            txt += '整体吸阻略偏重，可通过微调气道结构进一步优化。\n'
        elif 500 <= avg_loss < 700:
            txt += '整体吸阻明显，建议优化气道截面或减少局部收缩区域，以降低压降。\n'
        else:
            txt += '整体吸阻较大，需显著优化气道结构，如扩大截面、优化流线过渡等。\n'
        avg_loss_total = sum(total_pressure_losses) / len(total_pressure_losses)
        avg_loss_sensor = sum(sensor_pressure_losses) / len(sensor_pressure_losses)
        ratio = avg_loss_sensor / avg_loss_total * 100
        txt += f'咪头与出口的平均总压力损失为{avg_loss_sensor:.2f}Pa，占平均吸阻的比例为{ratio:.2f}\\%，'
        if ratio < 25:
            txt += (
                '启动灵敏。\n\n'
            )
        elif 25 <= ratio < 50:
            txt += (
                '启动灵敏度偏低。\n\n'
            )
        elif 50 <= ratio < 75:
            txt += (
                '压力损失较大，启动不灵敏。建议优化咪头与烟嘴的连接处设计。\n\n'
            )
        else:
            txt += (
                '压力损失过大，较难启动。强烈建议对咪头与烟嘴连接处进行大幅度调整，减小压力损失。\n\n'
            )
        return txt
