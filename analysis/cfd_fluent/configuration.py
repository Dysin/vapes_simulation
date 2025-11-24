'''
pyfluent 配置文件

Author: Dysin
Time:   2024.05.28
'''
import os.path

import ansys.fluent.core as pyfluent

class Fluent:
    def __init__(self, path, file_name, core_num=1, aoa=0, beta=0):
        self.core_num = core_num
        self.path = path
        self.file_name = file_name
        self.solver = None
        self.aoa = aoa
        self.beta = beta
    def launch(self, path_mesh, ui_mode=None):
        self.solver = pyfluent.launch_fluent(
            precision='double',
            processor_count=self.core_num,
            mode='solver',
            ui_mode=ui_mode
        )
        # pyfluent.search('<solver_session>.setup.boundary_conditions.velocity_inlet["<name>"].momentum.')
        self.solver.file.read(file_type='mesh', file_name=os.path.join(path_mesh, self.file_name + '.msh'))
        # pyfluent.search('boundary_condition')
        # self.solver.tui.mesh.check()
        # self.solver.tui.mesh.quality()
    def model(self, turbulence='sa'):
        if turbulence == 'sa':
            self.solver.setup.models.viscous = {'model': 'spalart-allmaras'}
        elif turbulence == 'k-omega':
            self.solver.setup.models.viscous = {"model": "k-omega", "k_omega_model": "sst"}
        else:
            print('Please enter the correct turbulence model!')
    def material_air(self, density, viscosity):
        self.solver.tui.define.materials.change_create(
            'air',
            'air',
            # change Density?
            'y',
            # mathods
            'constant',
            # value (in [kg/m^3])
            density,
            # change Cp (Specific Heat)?
            'n',
            # change Thermal Conductivity?
            'n',
            # change Viscosity?
            'y',
            # methods
            'constant',
            # value (in [kg/(m s)])
            viscosity,
            # change Molecular Weight?
            'n',
            # change Thermal Expansion Coefficient?
            'n',
            # change Speed of Sound?
            'n'
        )
    def bc_velocity_inlet(self, inlet_names, velocity, turb_intensity=5, turb_length_scale=1):
        for inlet_name in inlet_names:
            self.solver.tui.define.boundary_conditions.zone_type(inlet_name, 'velocity-inlet')
            velocity_inlet = self.solver.tui.define.boundary_conditions.set.velocity_inlet
            velocity_inlet(inlet_name, [], 'velocity-spec', 'n', 'y', 'quit')  # Components
            velocity_inlet(
                # bc name
                inlet_name, [],
                'direction-0',
                # Use Profile for X-Velocity?
                'n',
                # X-Velocity (constant or expression) (in [m/s])
                velocity[0],
                'direction-1',
                # Use Profile for Y-Velocity?
                'n',
                # Y-Velocity (constant or expression) (in [m/s])
                velocity[1],
                'direction-2',
                # Use Profile for Z-Velocity?
                'n',
                # Z-Velocity (constant or expression) (in [m/s])
                velocity[2],
                'quit'
            )
            velocity_inlet(inlet_name, [], 'ke-spec', 'n', 'y', 'quit') # Setting Intensity and Length Scale
            velocity_inlet(
                inlet_name, [],
                'turb-intensity',
                # Turbulent Intensity (constant or expression) (in [%])
                turb_intensity,
                # Turbulent Length Scale (constant or expression) (in [m])
                'turb-length-scale',
                turb_length_scale,
                'quit'
            )
        # help(velocity_inlet)
    def bc_pressure_outlet(self, outlet_names, direction, turb_viscosity_ratio=10):
        for outlet_name in outlet_names:
            self.solver.tui.define.boundary_conditions.zone_type(outlet_name, 'pressure-outlet')
            pressure_outlet = self.solver.tui.define.boundary_conditions.set.pressure_outlet
            pressure_outlet(outlet_name, [], 'direction-spec', 'y', 'quit')  # Direction-Vector
            pressure_outlet(
                outlet_name, [],
                'direction-0',
                # Use Profile for X-Component of Flow Direction? [no]
                'n',
                # X-Component of Flow Direction (constant or expression)
                direction[0],
                'direction-1',
                # Use Profile for Y-Component of Flow Direction? [no]
                'n',
                # Y-Component of Flow Direction (constant or expression)
                direction[1],
                'direction-2',
                # Use Profile for Z-Component of Flow Direction? [no]
                'n',
                # Z-Component of Flow Direction (constant or expression)
                direction[2],
                'quit'
            )
            pressure_outlet(outlet_name, [], 'ke-spec', 'n', 'n', 'y', 'quit')  # Turbulent Viscosity Ratio
            pressure_outlet(
                outlet_name, [],
                'turb-viscosity-ratio-profile',
                # Use Profile for Backflow Turbulent Viscosity Ratio?
                'n',
                # Backflow Turbulent Viscosity Ratio (constant or expression)
                turb_viscosity_ratio,
                'quit'
            )
    def bc_symmetry(self, symmetry_name):
        self.solver.tui.define.boundary_conditions.zone_type(symmetry_name, 'symmetry')
    # reference_values
    #   area:       参考面积 [m^2]
    #   density:    参考密度 [kg/m^3]
    #   viscosity:  参考粘度 [kg/(m·s)]
    #   length:     参考长度 [m]
    #   velocity:   参考速度 [m/s]
    def reference_values(self, area, density, viscosity, velocity, length=1):
        reference_values = self.solver.tui.report.reference_values
        reference_values.area(area)
        reference_values.density(density)
        reference_values.viscosity(viscosity)
        reference_values.velocity(velocity)
        reference_values.length(length)
        # reference_values.compute.velocity_inlet('', '', 'quit')
    def report_forces(self, wall_names, lift_vector, drag_vector, mom_vector, mom_center):
        # CL & Lift
        self.solver.tui.solve.report_definitions.add(
            'cl',
            'lift',
            'force-vector',
            lift_vector[0],
            lift_vector[1],
            lift_vector[2],
            'thread-names',
            wall_names
        )
        # self.solver.tui.solve.report_definitions('report-def-0', 'lift')
        self.solver.tui.solve.report_files.add(
            'cl',
            'report-defs',
            'cl',
            [],
            'print',
            'y',
            'file-name',
            '%s\lift_coefficient_aoa%s_beta%s.dat' %(self.path, self.aoa, self.beta)
        )
        self.solver.tui.solve.report_definitions.add(
            'lift',
            'force',
            'force-vector',
            lift_vector[0],
            lift_vector[1],
            lift_vector[2],
            'thread-names',
            wall_names
        )
        # self.solver.tui.solve.report_definitions('report-def-0', 'lift')
        self.solver.tui.solve.report_files.add(
            'lift',
            'report-defs',
            'lift',
            [],
            'print',
            'y',
            'file-name',
            '%s\lift_aoa%s_beta%s.dat' %(self.path, self.aoa, self.beta)
        )
        # CD & Drag
        self.solver.tui.solve.report_definitions.add(
            'cd',
            'drag',
            'force-vector',
            drag_vector[0],
            drag_vector[1],
            drag_vector[2],
            'thread-names',
            wall_names
        )
        # self.solver.tui.solve.report_definitions('report-def-0', 'lift')
        self.solver.tui.solve.report_files.add(
            'cd',
            'report-defs',
            'cd',
            [],
            'print',
            'y',
            'file-name',
            '%s\drag_coefficient_aoa%s_beta%s.dat' %(self.path, self.aoa, self.beta)
        )
        self.solver.tui.solve.report_definitions.add(
            'drag',
            'force',
            'force-vector',
            drag_vector[0],
            drag_vector[1],
            drag_vector[2],
            'thread-names',
            wall_names
        )
        # self.solver.tui.solve.report_definitions('report-def-0', 'lift')
        self.solver.tui.solve.report_files.add(
            'drag',
            'report-defs',
            'drag',
            [],
            'print',

            'y',
            'file-name',
            '%s\drag_aoa%s_beta%s.dat' %(self.path, self.aoa, self.beta)
        )
        # Cm
        self.solver.tui.solve.report_definitions.add(
            'cm',
            'moment',
            'mom-axis',
            mom_vector[0],
            mom_vector[1],
            mom_vector[2],
            'mom-center',
            mom_center[0],
            mom_center[1],
            mom_center[2],
            'thread-names',
            wall_names
        )
        # self.solver.tui.solve.report_definitions('report-def-0', 'lift')
        self.solver.tui.solve.report_files.add(
            'cm',
            'report-defs',
            'cm',
            [],
            'print',
            'y',
            'file-name',
            '%s\moment_coefficient_aoa%s_beta%s.dat' %(self.path, self.aoa, self.beta)
        )
    def residual(self, res=1e-6):
        self.solver.tui.solve.monitors.residual.convergence_criteria(
            res,
            res,
            res,
            res,
            res
        )
    def operation_conditions(self, pressure):
        self.solver.setup.general.operating_conditions.operating_pressure = pressure
    def initialize(self):
        self.solver.tui.solve.initialize.compute_defaults.all_zones()
        self.solver.solution.initialization.initialize()
    def auto_save(self, frequency):
        self.solver.file.write(file_name=self.path, file_type="case")
        self.solver.tui.file.auto_save.data_frequency(frequency)
        self.solver.tui.file.auto_save.root_name(self.path)
    def calculate(self, iteration_step):
        file_name = f'{self.file_name}_aoa{self.aoa}_beta{self.beta}'
        self.solver.tui.file.write_case(os.path.join(self.path, f'{file_name}.cas.h5'))
        self.solver.tui.solve.iterate(iteration_step)
        self.solver.tui.file.write_data(os.path.join(self.path, f'{file_name}.dat.h5'))
        self.solver.exit()

# if __name__ == '__main__':
#     file_name = 'aircraft_design_v0'
#     path_fluent = r'F:\project\aircraft\cfd\01_aircraft_design_v0\simulation\cfd_fluent'
#     cfd_fluent = Fluent(path_fluent, file_name, 2)
#     cfd_fluent.launch(ui_mode='gui')
#     cfd_fluent.model()
#     cfd_fluent.material_air(1.225, 1e-5)
#     cfd_fluent.bc_velocity_inlet(['inlet'], [33, 0, 0])
#     cfd_fluent.bc_pressure_outlet(['outlet'], [1, 0, 0])
#     cfd_fluent.bc_symmetry('symmetry')
#     cfd_fluent.reference_values(3, 1.2, 1e-5, 33)
#     cfd_fluent.report_forces(['wing_top', 'wing_bot', 'wing_tip'], [0, 0, 1], [1, 0, 0], [0, 1, 0], [0, 0, 0])
#     cfd_fluent.residual()
#     # cfd_fluent.initialize()
#     # cfd_fluent.calculate(200)
#     input()