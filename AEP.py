import random
import pyaedt
import torch

def dict_to_file(dict_data, file_name):
    """
    Writes a dictionary to a text file. Each key-value pair is written on a separate line, separated by a comma.
    
    Parameters:
        dict_data (dict): The dictionary to write to the file.
        file_name (str): The path to the file where the dictionary will be written.
    """
    with open(file_name, 'a') as file:
        for key, value in dict_data.items():
            file.write(f'{key},{value}\n')

def createVariation(hfss, positionlist=[0]):
    hfss['lambda']="30mm"
    hfss['z0']="0.5mm"
    hfss['subh']="1.57mm"
    hfss['copperh']="0.018mm"
    hfss['len1']="25mm"
    hfss['w1']="4.8mm"
    hfss['w2']="7.5mm"
    hfss['w3']="1.6mm"
    hfss['suby']="40mm"
    portnames=[]

    for i in range(17):
        if not i == 0 and not len(positionlist) == 17:
            positionlist.append(positionlist[i-1]+(random.uniform(15,30)))

        portrec = hfss.modeler.create_polyline(points=[(positionlist[i],0,'copperh'),(positionlist[i],0,'-subh-copperh')],
                                        name=f'port_{i}',
                                        xsection_orient='Y',
                                        xsection_type='Rectangle',
                                        xsection_width='w1',
                                        xsection_height='0',
                                        xsection_topwidth='0',
                                        )
        port = hfss.lumped_port(portrec, 
                        reference=None, 
                        create_port_sheet=False, 
                        port_on_plane=True, 
                        integration_line=hfss.AxisDir.ZPos, 
                        impedance=50
                        #name=f'port_{i}'
                        )

        portnames.append(port.name)
                                    
        antennline1=hfss.modeler.create_polyline(points=[(positionlist[i],0,'copperh/2'), (positionlist[i],'len1','copperh/2')],
                                                xsection_type='Rectangle',
                                                xsection_width='w1',
                                                xsection_height='copperh',
                                                name='line1',
                                                material='copper')
        antennaline2=hfss.modeler.create_polyline(points=[(positionlist[i]-13/2,25-7.5/2,'copperh/2'), (positionlist[i]+13/2,25-7.5/2,'copperh/2')],
                                                xsection_type='Rectangle',
                                                xsection_width='w2',
                                                xsection_height='copperh',
                                                name='line2',
                                                material='copper')
        antennaline3=hfss.modeler.create_polyline(points=[(positionlist[i]-13/2+1.6/2,15,'copperh/2'), (positionlist[i]-13/2+1.6/2,25,'copperh/2')],
                                                xsection_type='Rectangle',
                                                xsection_width='w3',
                                                xsection_height='copperh',
                                                name='line3',
                                                material='copper')                                    
        antennaline3.mirror(position=[positionlist[i],0,0],vector=[1,0,0],duplicate=True)
        antennaline3.unite([antennaline3.name+'_1',antennaline2,antennline1])

    hfss.modeler.create_polyline(points=[(-60,'suby/2','-subh/2'), (positionlist[-1]+60,'suby/2','-subh/2')],
                            xsection_type='Rectangle',
                            #xsection_orient='PositiveX',
                            xsection_width="suby",
                            #xsection_topwidth='0mm',
                            xsection_height="subh",
                            name="substrate",
                            material="Rogers RT/duroid 5880 (tm)")
    hfss.modeler.create_polyline(points=[(-60,'suby/2','-subh-copperh/2'), (positionlist[-1]+60,'suby/2','-subh-copperh/2')],
                                    xsection_type='Rectangle',
                                    xsection_width="suby",
                                    xsection_height="copperh",
                                    name="gnd",
                                    material="copper")

    return positionlist, portnames

def positionlist2positionDistribution(positionlist):
    positionDistribution = {}
    for i in range(17):
        if i == 0:
            positionDistribution[i] = (0,0,positionlist[i+1],positionlist[i+2]-positionlist[i+1])
        elif i == 1:
            positionDistribution[i] = (0,positionlist[i],positionlist[i+1]-positionlist[i],positionlist[i+2]-positionlist[i+1])
        elif i == 15:
            positionDistribution[i] = (positionlist[i-1]-positionlist[i-2],positionlist[i]-positionlist[i-1],positionlist[i+1]-positionlist[i],0)
        elif i == 16:
            positionDistribution[i] = (positionlist[i-1]-positionlist[i-2],positionlist[i]-positionlist[i-1],0,0)
        else:
            positionDistribution[i] = (positionlist[i-1]-positionlist[i-2],positionlist[i]-positionlist[i-1],positionlist[i+1]-positionlist[i],positionlist[i+2]-positionlist[i+1])
    return positionDistribution

def generateAEP(epoch=1):
    with pyaedt.Desktop(specified_version='241',non_graphical=True):
        for j in range(epoch):
            hfss = pyaedt.Hfss(projectname='17elements',designname='17elements',solution_type='Modal')
            hfss.logger.disabled = True
            hfss.create_open_region('10GHz')
            hfss.modeler.model_units = 'mm'
            setup = hfss.create_setup(name='a10',frequency='10GHz')
            setup.update()

            positionlist, portnames = createVariation(hfss)
            hfss.analyze(cores=8)
            positionDistribution = positionlist2positionDistribution(positionlist)
            rEPhiDic = {}
            for i in range(17):
                ccs = hfss.modeler.create_coordinate_system(origin=[positionlist[i], 0, 0], reference_cs='Global')
                infinite_sphere = hfss.insert_infinite_sphere(name='iis',x_start=-90, x_stop=90, x_step=1, y_start=0, y_stop=0, y_step=0, custom_coordinate_system=ccs.name)
                data = hfss.get_antenna_ffd_solution_data('10GHz',sphere=infinite_sphere.name)
                # rEPhi, rETheta, rETotal, Theta, Phi, nPhi, nTheta， Pincident, RealizedGain, RealizedGain_Total, RealizedGain_dB, RealizedGain_Theta, RealizedGain_Phi, Element_Location
                rEPhiDic[positionDistribution[i]] = pyaedt.modules.solutions.FfdSolutionData(frequencies=data.frequencies, eep_files=data.eep_files)._raw_data[f'{portnames[i]}_1']["rEPhi"]
                dict_to_file(rEPhiDic,'F:\\pythontxtfile\\eEPhi.txt')
                infinite_sphere.delete()
            print(f'##############################################################################################################################  Iteration {j} completed')
            if open('C:\\Users\\bacto\\Documents\\PyAEDT\\stop.txt').read() == '1':
                break

def validateAEP(positionlist):
    with pyaedt.Desktop(specified_version='241',non_graphical=True):
        hfss = pyaedt.Hfss(projectname='17elements',designname='17elements',solution_type='Modal')
        hfss.logger.disabled = True
        hfss.create_open_region('10GHz')
        hfss.modeler.model_units = 'mm'
        setup = hfss.create_setup(name='a10',frequency='10GHz')
        setup.update()

        createVariation(hfss,positionlist)
        hfss.analyze(cores=8)
        infinite_sphere = hfss.insert_infinite_sphere(name='iis',x_start=-90, x_stop=90, x_step=1, y_start=0, y_stop=0, y_step=0)
        data = hfss.get_antenna_ffd_solution_data('10GHz',sphere=infinite_sphere.name)
        # rEPhi, rETheta, rETotal, Theta, Phi, nPhi, nTheta， Pincident, RealizedGain, RealizedGain_Total, RealizedGain_dB, RealizedGain_Theta, RealizedGain_Phi, Element_Location
        rEPhiDic = pyaedt.modules.solutions.FfdSolutionData(frequencies=data.frequencies, eep_files=data.eep_files)._raw_data
        infinite_sphere.delete()
    return rEPhiDic

