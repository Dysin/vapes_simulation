'''
@Desc:   STAR-CCM+配置文件
@Author: Dysin
@Date:   2025/9/30
'''

import os
from textwrap import dedent

class StarCCMBuilder:
    def __init__(self, path):
        self.path = path.replace('\\', '\\\\')
    def header(self):
        '''
        生成Java宏头文件
        '''
        file = dedent(
            '''
            package macro;
            import java.util.*;
            import star.common.*;
            import star.base.neo.*;
            import star.segregatedflow.*;
            import star.material.*;
            import star.base.report.*;
            import star.turbulence.*;
            import star.vis.*;
            import star.flow.*;
            import star.kwturb.*;
            import star.metrics.*;
            import star.base.report.*;
            import star.prismmesher.*;
            import star.meshing.*;
            import star.acoustics.*;
            import star.trimmer.*;
            
            import star.segregatedenergy.*;
            import star.material.*;
            import star.electromagnetism.electricpotential.*;
            import star.electromagnetism.common.*;
            import star.electromagnetism.ohmicheating.*;
            import star.energy.*;
            
            public class mcr extends StarMacro {
            public void execute() {
            '''
        ).lstrip()
        return file

    def footer(self) -> str:
        '''生成 Java 宏结尾'''
        return dedent('}}')
    def write_mcrfile(self, file_name='mcr.java', content=''):
        '''
        输出Star-CCM+宏文件
        :param content: 文本内容
        :param file_name: 文件名
        :return:
        '''
        header = self.header()
        footer = self.footer()
        body = header + content + footer
        file_mcr = os.path.join(self.path, file_name)
        with open(file_mcr, 'w') as f:
            f.write(body)

    def base(self):
        file = dedent(
            '''
            Simulation simulation = getActiveSimulation();
            ImportManager importManager = simulation.getImportManager();
            '''
        ).lstrip()
        return file

    def units(self, unit):
        code = f'Units units_{unit} = ((Units) simulation.getUnitsManager().getObject("{unit}"));\n'
        return code

    def import_mesh(
            self,
            path,
            mesh_name
    ):
        path = path.replace('\\', '\\\\')
        file = dedent(
            f'''
            importManager.importMeshFiles(
            new StringVector(new String[] {{resolvePath("{path}\\\\{mesh_name}")}}),
            NeoProperty.fromString("{{'FileOptions': [{{'Sequence': 45}}]}}"));
            '''
        ).lstrip()
        return file

    def import_nas(self, path, geo_name, unit='mm'):
        path = path.replace('\\', '\\\\')
        file = dedent(
            f'''
            PartImportManager partImportManager = simulation.get(PartImportManager.class);
            partImportManager.importNastranPart(resolvePath("{path}\\\\{geo_name}.nas"), "OneSurfacePerPatch", "OnePartPerFile", true, units_{unit});
            '''
        ).lstrip()
        return file

    def scene(self, scene_name):
        code = (
            f'Scene scene_{scene_name} = simulation.getSceneManager().createScene("Repair Surface");'
        )
        return code

    def repair_surface(self, surface):
        code = (
            f'RootDescriptionSource rootDescriptionSource = simulation.get(SimulationMeshPartDescriptionSourceManager.class).getRootDescriptionSource();\n'
        )
        return code

    def get_mesh_part(self, part_name):
        code = (
            f'MeshPart meshPart_{part_name} = ((MeshPart) simulation.get(SimulationPartManager.class).getPart("{part_name}"));\n'
        )
        return code

    def get_part_surface(self, part_name, surface_name):
        code = (
            f'PartSurface partSurface_{surface_name} = ((PartSurface) meshPart_{part_name}.getPartSurfaceManager().getPartSurface("{surface_name}"));\n'
        )
        return code

    def automated_mesh(
            self,
            part_name,
            target_size,
            min_size,
            max_size,
            volume_growth_rate='FAST',
            bool_prism_layer=False,
            num_layers=None,
            layer_total_thickness=None,
            first_layer_thickness=None,
            layer_disable_boundary_names=None
    ):
        code = (
            f'simulation.getRegionManager().newRegionsFromParts(new ArrayList<>(Arrays.<GeometryPart>asList(meshPart_{part_name})), "OneRegionPerPart", null, "OneBoundaryPerPartSurface", null, RegionManager.CreateInterfaceMode.BOUNDARY, "OneEdgeBoundaryPerPart", null);\n'
        )
        if bool_prism_layer:
            code += (
                f'AutoMeshOperation autoMeshOperation_{part_name} = simulation.get(MeshOperationManager.class).createAutoMeshOperation(new StringVector(new String[] {{"star.resurfacer.ResurfacerAutoMesher", "star.resurfacer.AutomaticSurfaceRepairAutoMesher", "star.trimmer.TrimmerAutoMesher", "star.prismmesher.PrismAutoMesher"}}), new ArrayList<>(Arrays.<GeometryPart>asList(meshPart_{part_name})));\n'
                f'Collection<MesherBase> meshers = autoMeshOperation_{part_name}.getMeshersCollection();\n'
                f'for (MesherBase mesher : meshers) {{\n'
                f'    System.out.println("Mesher: " + mesher.getPresentationName());\n'
                f'}}\n'
                f'PrismAutoMesher prismAutoMesher_{part_name} = ((PrismAutoMesher) autoMeshOperation_{part_name}.getMeshers().getObject("Prism Layer Mesher"));\n'
                f'prismAutoMesher_{part_name}.getPrismStretchingOption().setSelected(PrismStretchingOption.Type.WALL_THICKNESS);\n'
                f'NumPrismLayers numPrismLayers_{part_name} = autoMeshOperation_{part_name}.getDefaultValues().get(PrismDefaultValuesManager.class).get(NumPrismLayers.class);\n'
                f'IntegerValue integerValue_{part_name} = numPrismLayers_{part_name}.getNumLayersValue();\n'
                f'integerValue_{part_name}.getQuantity().setValue({num_layers});\n'
                f'autoMeshOperation_{part_name}.getDefaultValues().get(PrismDefaultValuesManager.class).get(PrismWallThickness.class).setValueAndUnits({first_layer_thickness}, units_m);\n'
                f'PrismThickness prismThickness_{part_name} = autoMeshOperation_{part_name}.getDefaultValues().get(PrismDefaultValuesManager.class).get(PrismThickness.class);\n'
                f'prismThickness_{part_name}.getRelativeOrAbsoluteOption().setSelected(RelativeOrAbsoluteOption.Type.ABSOLUTE);'
                f'((ScalarPhysicalQuantity) prismThickness_{part_name}.getAbsoluteSizeValue()).setValue({layer_total_thickness});'
            )
            if layer_disable_boundary_names is not None:
                surfs = ','.join([
                    f'partSurface_{name}' for name in layer_disable_boundary_names
                ])
                print(surfs)
                code += (
                    f'SurfaceCustomMeshControl surfaceCustomMeshControl_{part_name} = autoMeshOperation_{part_name}.getCustomMeshControls().createSurfaceControl();\n'
                    f'PartsCustomizePrismMesh partsCustomizePrismMesh_{part_name} = surfaceCustomMeshControl_{part_name}.getCustomConditions().get(PartsCustomizePrismMesh.class );\n'
                    f'partsCustomizePrismMesh_{part_name}.getCustomPrismOptions().setSelected(PartsCustomPrismsOption.Type.DISABLE);\n'
                    f'surfaceCustomMeshControl_{part_name}.getGeometryObjects().setQuery(null);'
                    f'surfaceCustomMeshControl_{part_name}.getGeometryObjects().setObjects({surfs});\n'
                )
        else:
            code += (
                f'AutoMeshOperation autoMeshOperation_{part_name} = simulation.get(MeshOperationManager.class).createAutoMeshOperation(new StringVector(new String[] {{"star.resurfacer.ResurfacerAutoMesher", "star.resurfacer.AutomaticSurfaceRepairAutoMesher", "star.trimmer.TrimmerAutoMesher"}}), new ArrayList<>(Arrays.<GeometryPart>asList(meshPart_{part_name})));\n'
            )
        code += (
            f'PartsTargetSurfaceSize partsTargetSurfaceSize_{part_name} = autoMeshOperation_{part_name}.getDefaultValues().get(PartsTargetSurfaceSize.class);\n'
            f'partsTargetSurfaceSize_{part_name}.getRelativeOrAbsoluteOption().setSelected(RelativeOrAbsoluteOption.Type.ABSOLUTE);\n'
            f'((ScalarPhysicalQuantity) partsTargetSurfaceSize_{part_name}.getAbsoluteSizeValue()).setValue({target_size});\n'
            f'PartsMinimumSurfaceSize partsMinimumSurfaceSize_{part_name} = autoMeshOperation_{part_name}.getDefaultValues().get(PartsMinimumSurfaceSize.class);\n'
            f'partsMinimumSurfaceSize_{part_name}.getRelativeOrAbsoluteOption().setSelected(RelativeOrAbsoluteOption.Type.ABSOLUTE);\n'
            f'((ScalarPhysicalQuantity) partsMinimumSurfaceSize_{part_name}.getAbsoluteSizeValue()).setValue({min_size});\n'
            f'MaximumCellSize maximumCellSize_{part_name} = autoMeshOperation_{part_name}.getDefaultValues().get(MaximumCellSize.class);\n'
            f'maximumCellSize_{part_name}.getRelativeOrAbsoluteOption().setSelected(RelativeOrAbsoluteOption.Type.ABSOLUTE);\n'
            f'((ScalarPhysicalQuantity) maximumCellSize_{part_name}.getAbsoluteSizeValue()).setValue({max_size});\n'
            f'PartsSimpleTemplateGrowthRate partsSimpleTemplateGrowthRate_{part_name} = autoMeshOperation_{part_name}.getDefaultValues().get(PartsSimpleTemplateGrowthRate.class);\n'
            f'partsSimpleTemplateGrowthRate_{part_name}.getGrowthRateOption().setSelected(PartsGrowthRateOption.Type.{volume_growth_rate});\n'
            f'autoMeshOperation_{part_name}.executeSurfaceMeshers();\n'
            f'autoMeshOperation_{part_name}.execute();\n'
        )
        return code

    def generate_mesh(
            self,
            part_name,
            target_size,
            min_size,
            num_layers,
            growth_rate,
            prism_thickness,
            max_size
    ):
        file = dedent(
            f'''
            MeshPart meshPart_{part_name} = ((MeshPart) simulation.get(SimulationPartManager.class).getPart("{part_name}"));
            simulation.getRegionManager().newRegionsFromParts(new NeoObjectVector(new Object[] {{meshPart_{part_name}}}), "OneRegionPerPart", null, "OneBoundaryPerPartSurface", null, "OneFeatureCurve", null, RegionManager.CreateInterfaceMode.BOUNDARY);
            AutoMeshOperation autoMeshOperation_{part_name} = simulation.get(MeshOperationManager.class).createAutoMeshOperation(new StringVector(new String[] {{"star.resurfacer.ResurfacerAutoMesher", "star.resurfacer.AutomaticSurfaceRepairAutoMesher", "star.trimmer.TrimmerAutoMesher", "star.prismmesher.PrismAutoMesher"}}), new NeoObjectVector(new Object[] {{meshPart_{part_name}}}));
            for (Object mesherObj : autoMeshOperation_{part_name}.getMeshers().getObjects()) {{
            AutoMesher mesher = (AutoMesher) mesherObj;
            simulation.println("Available mesher: " + mesher.getPresentationName());
            }}
            // AutoMeshOperation autoMeshOperation_{part_name} = ((AutoMeshOperation) simulation.get(MeshOperationManager.class).getObject("Automated Mesh"));
            //PrismAutoMesher prismAutoMesher_{part_name} = ((PrismAutoMesher) autoMeshOperation_{part_name}.getMeshers().getObject("Prism Layer Mesher"));
            //autoMeshOperation_{part_name}.setLinkOutputPartName(false);
            //prismAutoMesher_{part_name}.getPrismStretchingOption().setSelected(PrismStretchingOption.Type.WALL_THICKNESS);
            PartsTargetSurfaceSize partsTargetSurfaceSize_{part_name} = autoMeshOperation_{part_name}.getDefaultValues().get(PartsTargetSurfaceSize.class);
            partsTargetSurfaceSize_{part_name}.getRelativeOrAbsoluteOption().setSelected(RelativeOrAbsoluteOption.Type.ABSOLUTE);
            ((ScalarPhysicalQuantity) partsTargetSurfaceSize_{part_name}.getAbsoluteSizeValue()).setValue({target_size});
            PartsMinimumSurfaceSize partsMinimumSurfaceSize_{part_name} = autoMeshOperation_{part_name}.getDefaultValues().get(PartsMinimumSurfaceSize.class);
            partsMinimumSurfaceSize_{part_name}.getRelativeOrAbsoluteOption().setSelected(RelativeOrAbsoluteOption.Type.ABSOLUTE);
            ((ScalarPhysicalQuantity) partsMinimumSurfaceSize_{part_name}.getAbsoluteSizeValue()).setValue({min_size});
            NumPrismLayers numPrismLayers_{part_name} = autoMeshOperation_{part_name}.getDefaultValues().get(NumPrismLayers.class);
            IntegerValue integerValue_{part_name} = numPrismLayers_{part_name}.getNumLayersValue();
            integerValue_{part_name}.getQuantity().setValue({num_layers});
             PrismLayerStretching prismLayerStretching_{part_name} = autoMeshOperation_{part_name}.getDefaultValues().get(PrismLayerStretching.class);
            prismLayerStretching_{part_name}.getStretchingQuantity().setValue({growth_rate});
            // autoMeshOperation_{part_name}.getDefaultValues().get(PrismWallThickness.class).setValue(--);
            PrismThickness prismThickness_{part_name} = autoMeshOperation_{part_name}.getDefaultValues().get(PrismThickness.class);
            prismThickness_{part_name}.getRelativeOrAbsoluteOption().setSelected(RelativeOrAbsoluteOption.Type.ABSOLUTE);
            ((ScalarPhysicalQuantity) prismThickness_{part_name}.getAbsoluteSizeValue()).setValue({prism_thickness});
            MaximumCellSize maximumCellSize_{part_name} = autoMeshOperation_{part_name}.getDefaultValues().get(MaximumCellSize.class);
            maximumCellSize_{part_name}.getRelativeOrAbsoluteOption().setSelected(RelativeOrAbsoluteOption.Type.ABSOLUTE);
            ((ScalarPhysicalQuantity) maximumCellSize_{part_name}.getAbsoluteSizeValue()).setValue({max_size});
            SurfaceCustomMeshControl surfaceCustomMeshControl_{part_name} = autoMeshOperation_{part_name}.getCustomMeshControls().createSurfaceControl();
            PartsCustomizePrismMesh partsCustomizePrismMesh_{part_name} = surfaceCustomMeshControl_{part_name}.getCustomConditions().get(PartsCustomizePrismMesh.class);
            partsCustomizePrismMesh_{part_name}.getCustomPrismOptions().setSelected(PartsCustomPrismsOption.Type.DISABLE);
            surfaceCustomMeshControl_{part_name}.getGeometryObjects().setQuery(null);
            PartSurface partSurface_inlet = ((PartSurface) meshPart_{part_name}.getPartSurfaceManager().getPartSurface("inlet"));
            PartSurface partSurface_outlet = ((PartSurface) meshPart_{part_name}.getPartSurfaceManager().getPartSurface("outlet"));
            surfaceCustomMeshControl_{part_name}.getGeometryObjects().setObjects(partSurface_inlet, partSurface_outlet);
            autoMeshOperation_{part_name}.executeSurfaceMeshers();
            autoMeshOperation_{part_name}.execute();
            '''
        ).lstrip()
        return file

    def generate_solid_mesh(
            self,
            part_name,
            target_size,
            min_size,
            max_size
    ):
        code = (
            f'MeshPart meshPart_{part_name} = ((MeshPart) simulation.get(SimulationPartManager.class).getPart("{part_name}"));\n'
            f'simulation.getRegionManager().newRegionsFromParts(new NeoObjectVector(new Object[] {{meshPart_{part_name}}}), "OneRegionPerPart", null, "OneBoundaryPerPartSurface", null, "OneFeatureCurve", null, RegionManager.CreateInterfaceMode.BOUNDARY);\n'
            f'AutoMeshOperation autoMeshOperation_{part_name} = simulation.get(MeshOperationManager.class).createAutoMeshOperation(new StringVector(new String[] {{"star.resurfacer.ResurfacerAutoMesher", "star.resurfacer.AutomaticSurfaceRepairAutoMesher", "star.trimmer.TrimmerAutoMesher"}}), new NeoObjectVector(new Object[] {{}}));\n'
            f'autoMeshOperation_{part_name}.setLinkOutputPartName(false);\n'
            f'PartsTargetSurfaceSize partsTargetSurfaceSize_{part_name} = autoMeshOperation_{part_name}.getDefaultValues().get(PartsTargetSurfaceSize.class);\n'
            f'partsTargetSurfaceSize_{part_name}.getRelativeSizeScalar().setValue({target_size});\n'
            f'PartsMinimumSurfaceSize partsMinimumSurfaceSize_{part_name} = autoMeshOperation_{part_name}.getDefaultValues().get(PartsMinimumSurfaceSize.class);\n'
            f'partsMinimumSurfaceSize_{part_name}.getRelativeSizeScalar().setValue({min_size});\n'
            f'MaximumCellSize maximumCellSize_{part_name} = autoMeshOperation_{part_name}.getDefaultValues().get(MaximumCellSize.class);\n'
            f'maximumCellSize_{part_name}.getRelativeSizeScalar().setValue({max_size});\n'
            f'autoMeshOperation_{part_name}.getInputGeometryObjects().setQuery(null);\n'
            f'autoMeshOperation_{part_name}.getInputGeometryObjects().setObjects(meshPart_{part_name});\n'
            f'autoMeshOperation_{part_name}.execute();\n'
        )
        return code

    def get_region(self, region_name):
        file = dedent(
            f'Region region_{region_name} = simulation.getRegionManager().getRegion("{region_name}");'
        )
        return file

    def get_boundary(self, region_name, boundary_name):
        code = (
            f'Boundary boundary_{boundary_name} = region_{region_name}.getBoundaryManager().getBoundary("{boundary_name}");\n'
        )
        return code

    def physics_rans_flow(
            self,
            physics_name='fluid',
            bool_aeroacoustics=True
    ):
        '''
        物理模型
        :param physics_name: 名称
        :param bool_aeroacoustics: 是否添加声学模型
        :return:
        '''
        code = dedent(
            f'''
            PhysicsContinuum physics_{physics_name} = ((PhysicsContinuum) simulation.getContinuumManager().getContinuum("Physics 1"));
            physics_{physics_name}.enable(SteadyModel.class);
            physics_{physics_name}.enable(SingleComponentGasModel.class);
            physics_{physics_name}.enable(SegregatedFlowModel.class);
            physics_{physics_name}.enable(ConstantDensityModel.class);
            physics_{physics_name}.enable(TurbulentModel.class);
            physics_{physics_name}.enable(RansTurbulenceModel.class);
            physics_{physics_name}.enable(KOmegaTurbulence.class);
            physics_{physics_name}.enable(SstKwTurbModel.class);
            physics_{physics_name}.enable(KwAllYplusWallTreatment.class);
            physics_{physics_name}.enable(ThreeDimensionalModel.class);
            physics_{physics_name}.setPresentationName("{physics_name}");
            '''
        ).lstrip()
        if bool_aeroacoustics:
            code += (
                f'physics_{physics_name}.enable(AcousticsTopModel.class );\n'
                f'physics_{physics_name}.enable(BroadbandAcousticsModel.class );\n'
                f'physics_{physics_name}.enable(BroadbandNoiseModel.class );\n'
                f'physics_{physics_name}.enable(CurleModel.class );\n'
                f'physics_{physics_name}.enable(ProudmanModel.class );\n'
            )
        return code

    def physics_electrothermal_solid(
            self,
            physics_name='solid',
            temporality='steady'
    ):
        code = (
            f'PhysicsContinuum physics_{physics_name} = ((PhysicsContinuum) simulation.getContinuumManager().getContinuum("Physics 1"));\n'
            # f'PhysicsContinuum physics_{physics_name} = simulation.getContinuumManager().createContinuum(PhysicsContinuum.class);\n'
            f'physics_{physics_name}.enable(ThreeDimensionalModel.class);\n'
            f'physics_{physics_name}.enable(SolidModel.class);\n'
            f'physics_{physics_name}.enable(ElectromagnetismModel.class);\n'
            f'physics_{physics_name}.enable(ElectrodynamicsPotentialModel.class);\n'
            f'physics_{physics_name}.enable(OhmicHeatingModel.class);\n'
            f'physics_{physics_name}.enable(SegregatedSolidEnergyModel.class);\n'
            f'physics_{physics_name}.enable(ConstantDensityModel.class);\n'
            f'physics_{physics_name}.setPresentationName("{physics_name}");\n'
        )
        if temporality == 'steady':
            code += f'physics_{physics_name}.enable(SteadyModel.class);\n'
        else:
            code += f'physics_{physics_name}.enable(ImplicitUnsteadyModel.class);\n'
        return code

    def material_solid(
            self,
            physics_name='solid',
            materail_name='materail',
            density=None,
            electrical_conductivity=None,
            specific_heat=None,
            thermal_conductivity=None,
    ):
        code = (
            f'SolidModel solidModel_{physics_name} = physics_{physics_name}.getModelManager().getModel(SolidModel.class);\n'
            f'Solid solid_{materail_name} = ((Solid) solidModel_{physics_name}.getMaterial());'
            f'solid_{materail_name}.setPresentationName("{materail_name}");\n'
            f'ConstantMaterialPropertyMethod constantMaterialPropertyMethod_density_{materail_name} = ((ConstantMaterialPropertyMethod) solid_{materail_name}.getMaterialProperties().getMaterialProperty(ConstantDensityProperty.class).getMethod());\n'
            f'constantMaterialPropertyMethod_density_{materail_name}.getQuantity().setValue({density});\n'
            f'ConstantMaterialPropertyMethod constantMaterialPropertyMethod_econductivity_{materail_name} = ((ConstantMaterialPropertyMethod) solid_{materail_name}.getMaterialProperties().getMaterialProperty(ElectricalConductivityProperty.class).getMethod());\n'
            f'constantMaterialPropertyMethod_econductivity_{materail_name}.getQuantity().setValue({electrical_conductivity});\n'
            f'ConstantSpecificHeat constantSpecificHeat_{materail_name} = ((ConstantSpecificHeat) solid_{materail_name}.getMaterialProperties().getMaterialProperty(SpecificHeatProperty.class).getMethod());\n'
            f'constantSpecificHeat_{materail_name}.getQuantity().setValue({specific_heat});\n'
            f'ConstantMaterialPropertyMethod constantMaterialPropertyMethod_tconductivity_{materail_name} = ((ConstantMaterialPropertyMethod) solid_{materail_name}.getMaterialProperties().getMaterialProperty(ThermalConductivityProperty.class).getMethod());\n'
            f'constantMaterialPropertyMethod_tconductivity_{materail_name}.getQuantity().setValue({thermal_conductivity});\n'
        )
        return code

    def boundary_pressure_outlet(self, boundary_name):
        code = (
            f'PressureBoundary pressureBoundary = ((PressureBoundary) simulation.get(ConditionTypeManager.class).get(PressureBoundary.class));\n'
            f'boundary_{boundary_name}.setBoundaryType(pressureBoundary);\n'
        )
        return code

    def boundary_stagnation_inlet(self, boundary_name, pressure):
        # 停滞压力
        code = (
            f'StagnationBoundary stagnationBoundary_{boundary_name} = ((StagnationBoundary) simulation.get(ConditionTypeManager.class).get(StagnationBoundary.class));\n'
            f'boundary_{boundary_name}.setBoundaryType(stagnationBoundary_{boundary_name});\n'
            f'TotalPressureProfile totalPressureProfile_{boundary_name} = boundary_{boundary_name}.getValues().get(TotalPressureProfile.class);\n'
            f'totalPressureProfile_{boundary_name}.getMethod(ConstantScalarProfileMethod.class).getQuantity().setValue({pressure});\n'
        )
        return code

    def boundary_mass_flow(self, boundary_name, mass_flow_rate):
        code = dedent(
            f'''
            MassFlowBoundary massFlowBoundary = ((MassFlowBoundary) simulation.get(ConditionTypeManager.class).get(MassFlowBoundary.class));
            boundary_{boundary_name}.setBoundaryType(massFlowBoundary);
            MassFlowRateProfile massFlowRateProfile = boundary_{boundary_name}.getValues().get(MassFlowRateProfile.class);
            massFlowRateProfile.getMethod(ConstantScalarProfileMethod.class).getQuantity().setValue({mass_flow_rate});
            '''
        ).lstrip()
        return code

    def boundary_wall_electrodynamics(
            self,
            boundary_name,
            electrical_potential_condition='insulator',
            thermal_condition='adiabatic',
            electric_potential=0,
            electrical_resistance=0,
            thermal_value=0
    ):
        '''
        电磁边界条件
        :param boundary_name: 边界名
        :param electric_potential: 电压值
        :param electrical_resistance: 电阻值
        :param electrical_potential_condition: 电势类型：electric_potential/insulator
        :param thermal_condition: 温度类型：adiabatic/heat_flux/heat_source/temperature
        :return:
        '''
        if electrical_potential_condition == 'electric_potential':
            code = (
                f'boundary_{boundary_name}.getConditions().get(ElectrodynamicsPotentialWallOption.class).setSelected(ElectrodynamicsPotentialWallOption.Type.ELECTRIC_POTENTIAL);\n'
                f'ElectricPotentialProfile electricPotentialProfile_{boundary_name} = boundary_{boundary_name}.getValues().get(ElectricPotentialProfile.class);\n'
                f'electricPotentialProfile_{boundary_name}.getMethod(ConstantScalarProfileMethod.class).getQuantity().setValue({electric_potential});\n'
                f'ElectricalResistanceAreaProfile electricalResistanceAreaProfile_{boundary_name} = boundary_{boundary_name}.getValues().get(ElectricalResistanceAreaProfile.class);\n'
                f'electricalResistanceAreaProfile_{boundary_name}.getMethod(ConstantScalarProfileMethod.class).getQuantity().setValue({electrical_resistance});\n'
            )
        else:
            code = (
                f'boundary_{boundary_name}.getConditions().get(ElectrodynamicsPotentialWallOption.class).setSelected(ElectrodynamicsPotentialWallOption.Type.INSULATOR);\n'
            )
        if thermal_condition == 'heat_flux':
            code += (
                f'boundary_{boundary_name}.getConditions().get(WallThermalOption.class).setSelected(WallThermalOption.Type.HEAT_FLUX);\n'
                f'HeatFluxProfile heatFluxProfile_{boundary_name} = boundary_{boundary_name}.getValues().get(HeatFluxProfile.class);'
                f'heatFluxProfile_{boundary_name}.getMethod(ConstantScalarProfileMethod.class).getQuantity().setValue({thermal_value});'
            )
        elif thermal_condition == 'heat_source':
            code += (
                f'boundary_{boundary_name}.getConditions().get(WallThermalOption.class).setSelected(WallThermalOption.Type.HEAT_SOURCE);\n'
            )
        elif thermal_condition == 'temperature':
            code += (
                f'boundary_{boundary_name}.getConditions().get(WallThermalOption.class).setSelected(WallThermalOption.Type.TEMPERATURE);\n'
            )
        else:
            code += f'boundary_{boundary_name}.getConditions().get(WallThermalOption.class).setSelected(WallThermalOption.Type.ADIABATIC);\n'
        return code

    def stopping_condition(
            self,
            type='steady',
            max_steps=1000,
            time_step=0.001,
            inner_iterations=5,
            max_time=None
    ):
        if type == 'steady':
            code = dedent(
                f'''
                StepStoppingCriterion stepStoppingCriterion = ((StepStoppingCriterion) simulation.getSolverStoppingCriterionManager().getSolverStoppingCriterion("Maximum Steps"));
                stepStoppingCriterion.setMaximumNumberSteps({max_steps});
                '''
            ).lstrip()
        else:
            if max_time is None:
                max_time = time_step * max_steps
            else:
                max_steps = max_time / time_step
            code = (
                f'ImplicitUnsteadySolver implicitUnsteadySolver = ((ImplicitUnsteadySolver) simulation.getSolverManager().getSolver(ImplicitUnsteadySolver.class));\n'
                f'implicitUnsteadySolver.getTimeStep().setValue({time_step});\n'
                # 二阶格式
                f'implicitUnsteadySolver.getTimeDiscretizationOption().setSelected(TimeDiscretizationOption.Type.SECOND_ORDER);\n'
                f'InnerIterationStoppingCriterion innerIterationStoppingCriterion = ((InnerIterationStoppingCriterion) simulation.getSolverStoppingCriterionManager().getSolverStoppingCriterion("Maximum Inner Iterations"));\n'
                f'innerIterationStoppingCriterion.setMaximumNumberInnerIterations({inner_iterations});\n'
                f'PhysicalTimeStoppingCriterion physicalTimeStoppingCriterion = ((PhysicalTimeStoppingCriterion) simulation.getSolverStoppingCriterionManager().getSolverStoppingCriterion("Maximum Physical Time"));\n'
                f'physicalTimeStoppingCriterion.getMaximumTime().setValue({max_time});\n'
                f'StepStoppingCriterion stepStoppingCriterion = ((StepStoppingCriterion) simulation.getSolverStoppingCriterionManager().getSolverStoppingCriterion("Maximum Steps"));\n'
                f'stepStoppingCriterion.setMaximumNumberSteps({max_steps});\n'
            )
        return code

    def create_monitor_and_plot(self, report_name, report_var):
        '''
        Create a monitor and plot the results.
        :param report_name:
        :param report_var:
        :return:
        '''
        # 2019版
        # file = dedent(
        #     f'''
        #     simulation.getMonitorManager().createMonitorAndPlot(new NeoObjectVector(new Object[] {{{report_var}_{report_name}}}), true, "%1$s Plot");
        #     ReportMonitor reportMonitor_{report_name} = ((ReportMonitor) simulation.getMonitorManager().getMonitor("{report_name} Monitor"));
        #     MonitorPlot monitorPlot_{report_name} = simulation.getPlotManager().createMonitorPlot(new NeoObjectVector(new Object[] {{reportMonitor_{report_name}}}), "{report_name} Monitor Plot");
        #     '''
        # ).lstrip()
        # 2025版
        code = (
            f'simulation.getMonitorManager().createMonitors(new ArrayList<>(Arrays.<Report>asList({report_var}_{report_name})), new PlotCreationInfo(PlotCreationInfo.CreatePlotChoice.SINGLE_PLOT, " Plot"));\n'
            f'Cartesian2DPlot cartesian2DPlot_{report_name} = ((Cartesian2DPlot) simulation.getPlotManager().getPlot("{report_name} Monitor Plot"));\n'
        )
        return code

    def get_field_function(self, var_name, function_name, function_type):
        '''
        获取场方程
        :param part_name: 变量名称
        :param function_name: 方程名
        :param function_type: 方程类型：scale/vector
        :return:
        '''
        if function_type == 'scale':
            file = dedent(
                f'''
                PrimitiveFieldFunction primitiveFieldFunction_{var_name} = ((PrimitiveFieldFunction) simulation.getFieldFunctionManager().getFunction("{function_name}"));
                '''
            ).lstrip()
        else:
            file = dedent(
                f'''
                PrimitiveFieldFunction primitiveFieldFunction_{var_name} = ((PrimitiveFieldFunction) simulation.getFieldFunctionManager().getFunction("{function_name}"));
                VectorMagnitudeFieldFunction vectorMagnitudeFieldFunction_{var_name} = ((VectorMagnitudeFieldFunction) primitiveFieldFunction_{var_name}.getMagnitudeFunction());
                VectorComponentFieldFunction vectorComponentFieldFunction_{var_name}0 = ((VectorComponentFieldFunction) primitiveFieldFunction_{var_name}.getComponentFunction(0));
                VectorComponentFieldFunction vectorComponentFieldFunction_{var_name}1 = ((VectorComponentFieldFunction) primitiveFieldFunction_{var_name}.getComponentFunction(1));
                VectorComponentFieldFunction vectorComponentFieldFunction_{var_name}2 = ((VectorComponentFieldFunction) primitiveFieldFunction_{var_name}.getComponentFunction(2));
                '''
            ).lstrip()
        return file

    def user_field_function(
            self,
            function_name,
            function_type,
            expression
    ):
        if function_type == 'scale':
            code = (
                f'UserFieldFunction userFieldFunction_{function_name} = simulation.getFieldFunctionManager().createFieldFunction();\n'
                f'userFieldFunction_{function_name}.getTypeOption().setSelected(FieldFunctionTypeOption.Type.SCALAR);\n'
                f'userFieldFunction_{function_name}.setDefinition("{expression}");\n'
                f'userFieldFunction_{function_name}.setPresentationName("{function_name}");\n'
            )
        else:
            code = ''
        return code

    def plane_section(
            self,
            section_name,
            direction,
            position,
            region_name
    ):
        if direction == 'x':
            coord = '1.0, 0.0, 0.0'
            pos = f'{position}, 0.0, 0.0'
        elif direction == 'y':
            coord = '0.0, 1.0, 0.0'
            pos = f'0.0, {position}, 0.0'
        elif direction == 'z':
            coord = '0.0, 0.0, 1.0'
            pos = f'0.0, 0.0, {position}'
        else:
            coord = '0.0, 0.0, 1.0'
            pos = f'0.0, 0.0, 0.0'
        code = (
            f'PlaneSection planeSection_{section_name} = (PlaneSection) simulation.getPartManager().createImplicitPart(new NeoObjectVector(new Object[] {{}}), new DoubleVector(new double[] {{0.0, 0.0, 1.0}}), new DoubleVector(new double[] {{0.0, 0.0, 0.0}}), 0, 1, new DoubleVector(new double[] {{0.0}}));\n'
            f'planeSection_{section_name}.getOrientationCoordinate().setCoordinate(units_m, units_m, units_m, new DoubleVector(new double[] {{{coord}}}));\n'
            f'planeSection_{section_name}.getOriginCoordinate().setCoordinate(units_m, units_m, units_m, new DoubleVector(new double[] {{{pos}}}));\n'
            f'planeSection_{section_name}.getInputParts().setQuery(null);\n'
            f'planeSection_{section_name}.getInputParts().setObjects(region_{region_name});\n'
            f'planeSection_{section_name}.setPresentationName("{section_name}");\n'
        )
        return code

    def report_surface_ave(
            self,
            function_name,
            function_type,
            part_name,
            part_type='boundary'
    ):
        '''
        计算面平均
        :param function_type: 方程类型：scale/vector
        :param function_name: 方程名
        :param part_name: part名
        :param part_type: part类型，boundary/planeSection
        :return:
        '''
        report_name = f'{function_name}_ave_{part_name}'
        if function_type == 'scale':
            report_var = 'primitiveFieldFunction'
        elif function_type == 'vector':
            report_var = 'vectorMagnitudeFieldFunction'
        else:
            report_var = 'userFieldFunction'
        code = (
            f'AreaAverageReport areaAverageReport_{report_name} = simulation.getReportManager().createReport(AreaAverageReport.class);\n'
            f'areaAverageReport_{report_name}.setFieldFunction({report_var}_{function_name});\n'
            f'areaAverageReport_{report_name}.getParts().setQuery(null);\n'
            f'areaAverageReport_{report_name}.getParts().setObjects({part_type}_{part_name});\n'
            f'areaAverageReport_{report_name}.setPresentationName("{report_name}");\n'
        )
        code += self.create_monitor_and_plot(report_name, 'areaAverageReport')
        return code, report_name

    def report_volume_ave(
            self,
            function_name,
            function_type,
            region_name
    ):
        report_name = f'{function_name}_ave_parts'
        if function_type == 'scale':
            report_var = 'primitiveFieldFunction'
        elif function_type == 'vector':
            report_var = 'vectorMagnitudeFieldFunction'
        else:
            report_var = 'userFieldFunction'
        code = (
            f'VolumeAverageReport volumeAverageReport_{report_name} = simulation.getReportManager().createReport(VolumeAverageReport.class);\n'
            f'volumeAverageReport_{report_name}.setFieldFunction({report_var}_{function_name});\n'
            f'volumeAverageReport_{report_name}.getParts().setQuery(null);\n'
            f'volumeAverageReport_{report_name}.getParts().setObjects(region_{region_name});\n'
            f'volumeAverageReport_{report_name}.setPresentationName("{report_name}");\n'
        )
        code += self.create_monitor_and_plot(
            report_name,
            'volumeAverageReport'
        )
        return code, report_name

    def report_max(
            self,
            function_name,
            function_type,
            parts,
            parts_type,
            bool_component=False,
            user_name = None
    ):
        '''
        监测最大值
        :param function_name: 函数名称
        :param function_type: 方程类型：scale/vector
        :param parts_type: parts类型：boundary/region/mix
        :param parts: part名，可以是单个或多个
        :param user_name: 自定义报告名称后缀
        :return: (生成的代码, 报告名称)
        '''
        # 统一处理 parts 为列表
        parts_list = [parts] if isinstance(parts, str) else parts
        # 生成 part_vars
        if parts_type == 'boundary':
            part_vars = [f'boundary_{name}' for name in parts_list]
        elif parts_type == 'region':
            part_vars = [f'region_{name}' for name in parts_list]
        else:  # mix 类型
            part_vars = parts_list
        # 生成报告名称
        report_suffix = user_name if user_name else 'parts' if len(parts_list) > 1 else parts_list[0]
        code = ''
        # 确定报告变量类型
        if function_type == 'scale':
            report_name = f'{function_name}_max_{report_suffix}'
            report_var = f'primitiveFieldFunction_{function_name}'
            # 生成代码
            part_vars_str = ', '.join(part_vars)
            code = (
                f'MaxReport maxReport_{report_name} = simulation.getReportManager().createReport(MaxReport.class);\n'
                f'maxReport_{report_name}.setFieldFunction({report_var});\n'
                f'maxReport_{report_name}.getParts().setQuery(null);\n'
                f'maxReport_{report_name}.getParts().setObjects({part_vars_str});\n'
                f'maxReport_{report_name}.setPresentationName("{report_name}");\n'
            )
            code += self.create_monitor_and_plot(report_name, 'maxReport')
            return code, report_name
        elif function_type == 'vector' and bool_component == False:
            report_name = f'{function_name}_max_{report_suffix}'
            report_var = f'vectorMagnitudeFieldFunction_{function_name}'
            # 生成代码
            part_vars_str = ', '.join(part_vars)
            code = (
                f'MaxReport maxReport_{report_name} = simulation.getReportManager().createReport(MaxReport.class);\n'
                f'maxReport_{report_name}.setFieldFunction({report_var});\n'
                f'maxReport_{report_name}.getParts().setQuery(null);\n'
                f'maxReport_{report_name}.getParts().setObjects({part_vars_str});\n'
                f'maxReport_{report_name}.setPresentationName("{report_name}");\n'
            )
            code += self.create_monitor_and_plot(report_name, 'maxReport')
            return code, report_name
        elif function_type == 'vector' and bool_component == True:
            report_names = []
            report_vars = []
            report_vars.append(f'vectorMagnitudeFieldFunction_{function_name}')
            report_vars.append(f'vectorComponentFieldFunction_{function_name}0')
            report_vars.append(f'vectorComponentFieldFunction_{function_name}1')
            report_vars.append(f'vectorComponentFieldFunction_{function_name}2')
            function_name_suffixs = ['', 'x', 'y', 'z']
            for i in range(len(report_vars)):
                report_name = f'{function_name}{function_name_suffixs[i]}_max_{report_suffix}'
                report_names.append(report_name)
                # 生成代码
                part_vars_str = ', '.join(part_vars)
                code += (
                    f'MaxReport maxReport_{report_name} = simulation.getReportManager().createReport(MaxReport.class);\n'
                    f'maxReport_{report_name}.setFieldFunction({report_vars[i]});\n'
                    f'maxReport_{report_name}.getParts().setQuery(null);\n'
                    f'maxReport_{report_name}.getParts().setObjects({part_vars_str});\n'
                    f'maxReport_{report_name}.setPresentationName("{report_name}");\n'
                )
                code += self.create_monitor_and_plot(report_name, 'maxReport')
            return code, report_names

    def report_min(
            self,
            function_name,
            function_type,
            parts,
            parts_type,
            bool_component=False,
            user_name = None
    ):
        '''
        监测最大值
        :param function_name: 函数名称
        :param function_type: 方程类型：scale/vector
        :param parts_type: parts类型：boundary/region/plane/mix
        :param parts: part名，可以是单个或多个
        :param user_name: 自定义报告名称后缀
        :return: (生成的代码, 报告名称)
        '''
        # 统一处理 parts 为列表
        parts_list = [parts] if isinstance(parts, str) else parts
        # 生成 part_vars
        if parts_type == 'boundary':
            part_vars = [f'boundary_{name}' for name in parts_list]
        elif parts_type == 'region':
            part_vars = [f'region_{name}' for name in parts_list]
        else:  # mix 类型
            part_vars = parts_list
        # 生成报告名称
        report_suffix = user_name if user_name else 'parts' if len(parts_list) > 1 else parts_list[0]
        code = ''
        # 确定报告变量类型
        if function_type == 'scale':
            report_name = f'{function_name}_min_{report_suffix}'
            report_var = f'primitiveFieldFunction_{function_name}'
            # 生成代码
            part_vars_str = ', '.join(part_vars)
            code = (
                f'MinReport minReport_{report_name} = simulation.getReportManager().createReport(MinReport.class);\n'
                f'minReport_{report_name}.setFieldFunction({report_var});\n'
                f'minReport_{report_name}.getParts().setQuery(null);\n'
                f'minReport_{report_name}.getParts().setObjects({part_vars_str});\n'
                f'minReport_{report_name}.setPresentationName("{report_name}");\n'
            )
            code += self.create_monitor_and_plot(report_name, 'minReport')
            return code, report_name
        elif function_type == 'vector' and bool_component == False:
            report_name = f'{function_name}_min_{report_suffix}'
            report_var = f'vectorMagnitudeFieldFunction_{function_name}'
            # 生成代码
            part_vars_str = ', '.join(part_vars)
            code = (
                f'MinReport minReport_{report_name} = simulation.getReportManager().createReport(MinReport.class);\n'
                f'minReport_{report_name}.setFieldFunction({report_var});\n'
                f'minReport_{report_name}.getParts().setQuery(null);\n'
                f'minReport_{report_name}.getParts().setObjects({part_vars_str});\n'
                f'minReport_{report_name}.setPresentationName("{report_name}");\n'
            )
            code += self.create_monitor_and_plot(report_name, 'minReport')
            return code, report_name
        elif function_type == 'vector' and bool_component == True:
            report_names = []
            report_vars = []
            report_vars.append(f'vectorMagnitudeFieldFunction_{function_name}')
            report_vars.append(f'vectorComponentFieldFunction_{function_name}0')
            report_vars.append(f'vectorComponentFieldFunction_{function_name}1')
            report_vars.append(f'vectorComponentFieldFunction_{function_name}2')
            function_name_suffixs = ['', 'x', 'y', 'z']
            for i in range(len(report_vars)):
                report_name = f'{function_name}{function_name_suffixs[i]}_min_{report_suffix}'
                report_names.append(report_name)
                # 生成代码
                part_vars_str = ', '.join(part_vars)
                code += (
                    f'MinReport minReport_{report_name} = simulation.getReportManager().createReport(MinReport.class);\n'
                    f'minReport_{report_name}.setFieldFunction({report_vars[i]});\n'
                    f'minReport_{report_name}.getParts().setQuery(null);\n'
                    f'minReport_{report_name}.getParts().setObjects({part_vars_str});\n'
                    f'minReport_{report_name}.setPresentationName("{report_name}");\n'
                )
                code += self.create_monitor_and_plot(report_name, 'minReport')
            return code, report_names

    def report_expression(
            self,
            report_name,
            expression
    ):
        code = (
            f'ExpressionReport expressionReport_{report_name} = simulation.getReportManager().createReport(ExpressionReport.class);\n'
            f'expressionReport_{report_name}.setDefinition("{expression}");\n'
            f'expressionReport_{report_name}.setPresentationName("{report_name}");\n'
        )
        code += self.create_monitor_and_plot(report_name, 'expressionReport')
        return code

    def report_export(self, report_name):
        # # 2019版
        # file = dedent(
        #     f'''
        #     monitorPlot_{report_name}.export(resolvePath("{self.path}\\\\{report_name}.csv"), ",");
        #     '''
        # ).lstrip()
        # 2025版
        code = (
            f'cartesian2DPlot_{report_name}.export(resolvePath("{self.path}\\\\{report_name}.csv"), ",");\n'
        )
        return code

    def initialize(self):
        file = dedent(
            '''
            Solution solution = simulation.getSolution();
            solution.initializeSolution();
            '''
        ).lstrip()
        return file

    def run(self):
        file = dedent(
            '''
            simulation.getSimulationIterator().run();
            '''
        ).lstrip()
        return file

    def save(self, file_name):
        file = dedent(
            f'''
            simulation.saveState("{self.path}\\\\{file_name}.sim");
            '''
        ).lstrip()
        return file

    def export_tecplot(self, file_name, region_name, functions):
        functions_string = ', '.join(str(x) for x in functions)  # 列表转换为单个字符串
        file = dedent(
            f'''
            importManager.setExportPath(\"{self.path}\\\\{file_name}.plt\");
            importManager.setFormatType(SolutionExportFormat.Type.PLT);
            importManager.setExportParts(new NeoObjectVector(new Object[] {{}}));
            importManager.setExportPartSurfaces(new NeoObjectVector(new Object[] {{}}));
            importManager.setExportBoundaries(new NeoObjectVector(new Object[] {{}}));
            importManager.setExportRegions(new NeoObjectVector(new Object[] {{region_{region_name}}}));
            importManager.setExportScalars(new NeoObjectVector(new Object[] 
            {{{functions_string}}}));
            importManager.setExportVectors(new NeoObjectVector(new Object[] {{}}));
            importManager.setExportOptionAppendToFile(false);
            importManager.setExportOptionDataAtVerts(false);
            importManager.setExportOptionSolutionOnly(false);
            importManager.export(resolvePath(\"{self.path}\\\\{file_name}.plt\"), new NeoObjectVector(new Object[] {{region_{region_name}}}), new NeoObjectVector(new Object[] {{}}), new NeoObjectVector(new Object[] {{}}), new NeoObjectVector(new Object[] {{}}), new NeoObjectVector(new Object[] {{{functions_string}}}), NeoProperty.fromString("{{\'exportFormatType\': 6, \'appendToFile\': false, \'solutionOnly\': false, \'dataAtVerts\': false}}"));
            '''
        ).lstrip()
        return file

if __name__ == '__main__':
    path = r'D:\1_Work\templates\vapes_simulation\test_case\simulation\starccm_flow_steady'
    ccm_builder = StarCCMBuilder(path)
    txt = ''
    txt += ccm_builder.import_mesh(path, mesh_name='demo.msh')
    txt += ccm_builder.physics_rans_flow()
    txt += ccm_builder.get_region(region_name='fluid')
    txt += ccm_builder.boundary_pressure(region_name='fluid', boundary_name='inlet')
    txt += ccm_builder.boundary_mass_flow(region_name='fluid', boundary_name='outlet', mass_flow_rate=0.1)
    txt += ccm_builder.stopping_condition(max_steps=100)
    txt += ccm_builder.report_surface_ave('inlet', 'Pressure', 'p_in')
    txt += ccm_builder.report_surface_ave('outlet', 'Pressure', 'p_out')
    txt += ccm_builder.report_expression(
        '${p_out} - ${p_in}',
        'delta_p'
    )
    txt += ccm_builder.initialize()
    txt += ccm_builder.run()
    txt += ccm_builder.report_export('p_in')
    txt += ccm_builder.report_export('p_out')
    txt += ccm_builder.report_export('delta_p')
    ccm_builder.write_mcrfile(content=txt)