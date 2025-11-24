'''
湍流参数
Author: Dysin
Time:   2024.05.23
'''

import math

class Turbulence_Params():
    R = 287.058
    gamma = 1.4
    c_mu = 0.09  # 湍流参数
    def __init__(self, velocity_ref, l_ref):
        self.t = 273.15 + 20
        self.p = 101325
        self.k = 0.0262
        self.c = 343
        self.rho = 1.204
        self.velocity_ref = velocity_ref
        self.l_ref = l_ref

    def info(self):
        # 打印结果
        print(f"Speed of Sound:\t\t\t{self.c} m/s")
        print(f"Density:\t\t\t\t{self.rho} kg/m^3")
        print(f"Pressure:\t\t\t\t{self.p} Pa")
        print(f"Temperature:\t\t\t{self.t} K")
        print(f"Thermal Conductivity:\t{self.k} W/(m*K)")

    # 根据Sutherland公式计算动力粘度 [Pa·s]
    def mu(self):
        mu_ref = 1.4584e-6
        t_ref = 110.33
        result = mu_ref * math.sqrt(self.t) / (t_ref / self.t + 1.0)
        return result

    def nu(self):
        return self.mu() / self.rho

    # 雷诺数
    def reynolds_number(self):
        result = self.rho * self.velocity_ref * self.l_ref / self.mu()
        return result

    # 当地声速
    def acoustic_velociy(self):
        result = math.pow(self.gamma * self.R * self.t, 2)
        return result

    # 湍流强度
    def turbulence_intensity(self):
        re = self.reynolds_number()
        result = 0.16 * math.pow(re, -0.125)
        return result

    # 湍流长度尺度 [m]
    def turbulence_length_scale(self):
        result = 0.07 * self.l_ref / math.pow(self.c_mu, 0.75)
        return result

    # 湍动能 [(m^2/s^2]
    def turbulence_kinetic_energy(self):
        turbulence_intensity = self.turbulence_intensity()
        result = 1.5 * math.pow(turbulence_intensity * self.velocity_ref, 2)
        return result

    # 湍流耗散率 [m^2/s^3]
    def turbulence_epsilon(self, velocity, l_ref):
        turbulence_kinetic_energy = self.turbulence_kinetic_energy()
        turbulence_length_scale = self.turbulence_length_scale()
        result = math.pow(turbulence_kinetic_energy, 1.5) / turbulence_length_scale
        return result

    # 湍流粘度比
    def turbulence_viscosity_rate(self):
        mut = math.sqrt(1.5) * self.c_mu * self.velocity_ref * self.turbulence_intensity() * self.turbulence_length_scale()
        result = mut / self.mu()
        return result

    # 比耗散率 [1/s]
    def turbulence_omega(self):
        result = self.rho * self.turbulence_kinetic_energy() / (self.mu() * self.turbulence_viscosity_rate())
        return result

    # 根据状态方程计算压力 [Pa]
    def pressure(self):
        result = self.rho * self.R * self.t
        return result

    def boundary_layer_first_y(self, y_plus, cf_model='schlichting'):
        '''
        计算第一层网格高度 y1
        :param y_plus: 目标 y+
        :param cf_model: 'schlichting' (Cf=0.026 Re^{-1/7}) 或 'blasius' 等
        :return: y1 (m)
        '''
        Re = self.reynolds_number()
        nu = self.nu()
        U = self.velocity_ref

        # 选择 Cf 经验式（可扩展）
        if cf_model == 'schlichting':
            Cf = 0.026 * Re ** (-1.0 / 7.0)
        elif cf_model == 'empirical_flatplate':
            Cf = 0.058 * Re ** (-1.0 / 5.0)  # 举例，按需替换
        else:
            # 默认回退：使用 schlichting
            Cf = 0.026 * Re ** (-1.0 / 7.0)

        u_tau = U * math.sqrt(Cf / 2.0)
        if u_tau <= 0:
            raise ValueError("Computed friction velocity non-positive")

        y1 = y_plus * nu / u_tau
        print(f"[INFO] Re={Re:.3e}, Cf={Cf:.3e}, u_tau={u_tau:.3e}, y1={y1:.3e} m")
        return y1

    def boundary_layer_growth_rate(self, y_plus, last_layer_height, n_layers):
        '''
        已知第一层与最后一层高度、层数，求增长率
        :param first_layer_height: 第一层网格高度
        :param last_layer_height: 最后一层网格高度
        :param n_layers: 边界层层数
        :return: 增长率 r
        '''
        first_layer_height = self.boundary_layer_first_y(y_plus)
        if n_layers <= 1:
            raise ValueError("层数必须大于1")
        r = (last_layer_height / first_layer_height) ** (1 / (n_layers - 1))
        print(f'[INFO] Boundary layer growth rate：{r}')
        return r

    def prism_thickness(self, y_plus, growth_rate, n_layers):
        '''
        计算棱柱层总厚度
        :param y_plus: y+
        :param growth_rate: 增长率
        :param n_layers: 层数
        :return: 棱柱层总厚度
        '''
        first_layer_height = self.boundary_layer_first_y(y_plus)
        if n_layers < 1:
            raise ValueError("层数必须大于等于1")
        if abs(growth_rate - 1.0) < 1e-6:
            # 特殊情况：等厚
            total = first_layer_height * n_layers
        else:
            total = first_layer_height * (math.pow(growth_rate, n_layers) - 1) / (growth_rate - 1)
        print(f'[INFO] Boundary layer total thickness: {total}')
        return total

if __name__ == '__main__':
    turbulence = Turbulence_Params(velocity_ref=0.46, l_ref=3e-3)
    first_y = turbulence.boundary_layer_first_y(1)
    r = turbulence.boundary_layer_growth_rate(1, 6e-4, 5)
    turbulence.prism_thickness(1, r, 5)
    print(first_y, r)