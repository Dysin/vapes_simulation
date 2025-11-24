'''
@Desc:   单位转换工具
@Author: Dysin
@Date:   2025/10/10
'''

class UnitManager:
    '''
    通用单位转换类
    '''
    def __init__(self, density=1.225):
        self.density = density

    def convert(self, value, from_unit, to_unit, area=None):
        """
        转换空气流动单位：mL/s、kg/s、m/s 之间的换算。

        参数：
            value : float
                输入的数值
            from_unit : str
                原单位 ('mL/s', 'kg/s', 'm/s')
            to_unit : str
                目标单位 ('mL/s', 'kg/s', 'm/s')
            density : float, optional
                空气密度 (kg/m³)，默认 1.204 (20°C, 1atm)
            area : float, optional
                截面积 (m²)，当涉及速度(m/s)转换时必须提供

        返回：
            float : 转换后的数值
        """
        # Step 1: 转换为标准单位 m³/s
        if from_unit == 'mL/s':
            m3_s = value * 1e-6
        elif from_unit == 'kg/s':
            m3_s = value / self.density
        elif from_unit == 'm/s':
            if area is None:
                raise ValueError("需要提供截面积 area 来进行速度转换")
            m3_s = value * area
        else:
            raise ValueError(f"未知单位: {from_unit}")

        # Step 2: 从 m³/s 转换到目标单位
        if to_unit == 'mL/s':
            return m3_s * 1e6
        elif to_unit == 'kg/s':
            return m3_s * self.density
        elif to_unit == 'm/s':
            if area is None:
                raise ValueError("需要提供截面积 area 来进行速度转换")
            return m3_s / area
        else:
            raise ValueError(f"未知单位: {to_unit}")


if __name__ == '__main__':
    unit = UnitManager()
    res = unit.convert(17.5, from_unit='mL/s', to_unit='kg/s')
    print(res)
    res = unit.convert(17.5, from_unit='mL/s', to_unit='m/s', area=37.9e-6)
    print(res)