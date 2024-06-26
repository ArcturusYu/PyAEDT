import pyaedt
# with pyaedt.Desktop(specified_version='221',non_graphical=False):
#     hfss=pyaedt.Hfss(projectname="C:\\Users\\bacto\\Documents\\Ansoft\\Ka_L2-liyus.aedt",designname='lili',solution_type='Modal')
#     hfss['tt']#上面田字形的耦合框框，越大空窗越小
#     hfss['lp']#上面的耦合片片边长
hfss=pyaedt.Hfss(projectname="C:\\Users\\bacto\\Documents\\Ansoft\\Ka_L2-liyu.aedt",
                 designname='HFSSDesign2',
                 solution_type='Modal',
                 specified_version='221',
                 non_graphical=False)

hfss['tt']='0.26mm'#上面田字形的耦合框框，越大空窗越小
hfss['lp']='3.605mm'#上面的耦合片片边长
# hfss['rectlong']='3.4mm'#下面的耦合片片长边
# hfss['rectshort']='0.5mm'#下面的耦合片片短边
hfss['copperh']='0.018mm'#金属层厚度
# hfss.modeler.coordinate_systems()
# rect = hfss.modeler.create_polyline(points=[[0,'lf-rectshort/2','hs+copperh/2'],[0,'lf+rectshort/2','hs+copperh/2']],
#                                     xsection_type='Rectangle',
#                                     xsection_width='rectlong',
#                                     xsection_height='copperh',
#                                     name='line1',
#                                     material='copper')
# rect.mirror(position=[0,0,0],vector=[-1,-1,0],duplicate=True)
# rect.unite(rect.name+'_1')
# rect.duplicate_around_axis(axis='Z',clones=4,create_new_objects=False)
# rect.subtract('Feed_in1')

para=hfss.parametrics