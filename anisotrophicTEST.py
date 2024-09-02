import pyaedt
import numpy as np
import pyaedt.modeler
from pyaedt.modules.Boundary import BoundaryProps
#调整复材的三维
x = 100
y = 100
z = 100
#复材的各向异性介电常数，simple模式，也可以是一个张量
anisotrophicDielectric = [1,1,1]

variables = [x, y, z, anisotrophicDielectric]
keys = ['x', 'y', 'z', 'anisotrophicDielectric']

desktop = pyaedt.Desktop(version=232, non_graphical=False)
hfss = pyaedt.Hfss(solution_type='Modal')
hfss.logger.disabled = True
hfss.logger.disable_desktop_log()
hfss.logger.disable_stdout_log()
hfss.modeler.model_units = 'mm'
setup = hfss.create_setup(name='a10')
# 1-6g 50m
setup.create_frequency_sweep(start_frequency='1GHz', stop_frequency='6GHz', num_of_freq_points=101)
setup.update()
hfss.materials.add_material('complexmaterial')
hfss.materials['complexmaterial'].permittivity = anisotrophicDielectric

#这里设置了个region，上下两个面是波端口，100相当于波端口到复材表面的距离
region = hfss.modeler.create_region(pad_value=[0,0,0,0,100,100], pad_type='Absolute Offset')
box = hfss.modeler.create_box([0, 0, 0], [x, y, z], name='box', material='complexmaterial')
#对复材的XY平面设置lattice pairs
hfss.auto_assign_lattice_pairs(assignment = box, coordinate_plane = 'XY')
# hfss.assign_radiation_boundary_to_faces([region.bottom_face_z, region.top_face_z])

#设置波端口，每个端口设置两个模式
port0 = hfss.wave_port(region.bottom_face_z, name='port0', modes=2, integration_line=hfss.AxisDir.XPos)
propIntline = {
    'Start': [f'{x/2}mm', f'{y}mm', '-100.0mm'],
    'End': [f'{x/2}mm', '0.0mm', '-100.0mm']
}
boundary_props_Intline = BoundaryProps(port0, propIntline)
propMode = {
    'ModeNum': 2,
    'UseIntLine': True,
    'IntLine': boundary_props_Intline,
    'AlignmentGroup': 0,
    'CharImp': 'Zpi',
    'RenormImp': '50ohm'
}
boundary_props_Mode = BoundaryProps(port0, propMode)
port0['Modes/Mode2'] = boundary_props_Mode

portz = hfss.wave_port(region.top_face_z, name='portz', modes=2, integration_line=hfss.AxisDir.XPos)
propIntline = {
    'Start': [f'{x/2}mm', f'{y}mm', f'{z+100.0}mm'],
    'End': [f'{x/2}mm', '0.0mm', f'{z+100.0}mm']
}
boundary_props_Intline = BoundaryProps(portz, propIntline)
propMode = {
    'ModeNum': 2,
    'UseIntLine': True,
    'IntLine': boundary_props_Intline,
    'AlignmentGroup': 0,
    'CharImp': 'Zpi',
    'RenormImp': '50ohm'
}
boundary_props_Mode = BoundaryProps(portz, propMode)
portz['Modes/Mode2'] = boundary_props_Mode


validation = hfss.analyze(cores=8)
solutiondata = hfss.post.get_solution_data(expressions='S(port0:2,port0:1)')
solutiondata.data_db10('S(port0:2,port0:1)')