'''
Date: 15/11/2023


'''

import os
os.chdir(os.path.dirname(__file__))    #path della cartella che contiene il progetto
#print(os.getcwd())

from run import *



## main di prova ##

# run1 = Run()
# file_name = "../Dataset/Matilde_13_09_2023_AM.csv"
# run1.readRun(file_name)
# # print(run1.run_data["speed"])
# # print(run1.run_data["cadence"])

# gb1 = GearBox([40,35,31,27,24,21,19,17,15,14,13,12])
# gb1.setInfo(chainring=108, sec_ratio=[38,18])
# wl1 = Wheel("Michelin-blue")
# wl1.setInfo(None,0.23157,None,None)
# bk1 = BikeInfo(Vehicle("Phoenix"),Driver("Matilde"),wl1,gb1)
# run1.setBikeInfo(bk1)
# # print(run1.run_data.columns.values)
# # run1.setBounds(15,30)
# run1.gearChangeDetect()
# # print(run1.run_data[['gear','speed']])
# run1.exportCols("prova1.csv", ['gear', 'speed'])
# # print(run1.run_data.iloc[0:7][['speed','gear','RPMw_bo_RPMp','ideal_speed']])
# # print(run1.run_data[['speed','gear','cadence','RPMw_bo_RPMp','ideal_speed']])
# # print(run1.disp)
# # new_power = run1.generatePower()
# # run1.Modeling(1)
# # run1.Simulation()
# run1.plot()
# run1.export()


run2 = Run()
file_name = "../Dataset/Diego_15_09_2023_AM_2.csv"
run2.readRun(file_name)
# print(run1.run_data["speed"])
# print(run1.run_data["cadence"])

gb2 = GearBox([32,28,24,21,19,17,15])
gb2.setInfo(chainring=60, sec_ratio=[54,17])
wl2 = Wheel("Michelin-blue")
wl2.setInfo(None,0.23157,None,None)
bk2 = BikeInfo(Vehicle("Cerberus"),Driver("Diego"),wl2,gb2)
run2.setBikeInfo(bk2)
# print(run2.run_data.columns.values)
# run1.setBounds(15,30)
run2.gearChangeDetect()
# print(run2.run_data[['gear','speed']])
# run2.exportCols("prova2.csv", ['gear', 'speed'])
# print(run2.run_data.iloc[0:7][['speed','gear','RPMw_bo_RPMp','ideal_speed']])
# print(run2.run_data[['speed','gear','cadence','RPMw_bo_RPMp','ideal_speed']])
# print(run2.disp)
# run2.export()

run3 = Run()
file_name = "../Dataset/Diego_13_09_2023_AM_2.csv"
run3.readRun(file_name)

gb3 = GearBox([32,28,24,21,19,17,15])
gb3.setInfo(chainring=60, sec_ratio=[54,17])
wl3 = Wheel("Michelin-blue")
wl3.setInfo(None,0.23157,None,None)
bk3 = BikeInfo(Vehicle("Cerberus"),Driver("Diego"),wl3,gb3)
run3.setBikeInfo(bk3)
# print(run3.run_data.columns.values)
# run3.setBounds(15,30)
run3.gearChangeDetect()



an_run = RunAnalysis()
an_run.addRun(run3)
an_run.addRun(run2)
print(an_run.run_list.keys())
# an_run.rmRun("prova")
# print(an_run.run_list.keys())
# an_run.plotEach()
# an_run.comparation(cols=["power"]) #keys=["Diego_15_09_2023_AM_2.csv","Diego_13_09_2023_AM_2.csv"],cols=["speed","ideal_speed","power"])
# an_run.plotEach(export=True)
an_run.modeling(input_values=["power","heart_rate"],plot=True)
an_run.simulate(plot=True)
an_run.run_list["avg_run"].plot()
an_run.run_list["sim_run"].plot()

