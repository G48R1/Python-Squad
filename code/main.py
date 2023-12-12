'''
Date: 15/11/2023
cosa fa questo codice?
prende i dati delle run (che sono salvati in dei file csv) e li rielabora per produrre dei valori
e dei grafici significativi e utili per l'analisi della run.
È un codice abbastanza versatile, e permette di avere molta flessibilità sugli output che si vogliono
visualizzare.
Per un oggetto RunAnalysis si possono selezionare manualmente i dati che si intende visulizzare,
già divisi per plot (ad esempio, si possono rappresentare i profili di "power" e "speed" in una finestra,
mentre la "speed" e la "ideal_speed" in un'altra).
Si possono anche selezionare delle opzioni (opts) preimpostate, che corrispondono a dei blocchi di grafici
predefiniti, associati alle preferenze usuali dei ciclisti e/o di chi analizzerà i dati.
C'è anche un modulo di simulazione, ma è ancora da sistemare.
Però nel frattempo tu...
crea una copia di questo codice nel tuo PC e divertiti a cambiare i parametri a tuo piacimento!

N.B. i path utilizzati in questo codice sono path relativi. Ricordati di non spostare i file, altrimenti
nulla funzionerà più :)
'''
from run import *
import os
os.chdir(os.path.dirname(__file__))    #path della cartella che contiene il progetto
#print(os.getcwd())


## main di prova ##

dataset_path = "../Dataset/"
conditions_path = "../Dataset/conditions/"

an_run = RunAnalysis()

run1 = Run()
file_name = dataset_path + "Matilde_13_09_2023_AM.csv"
run1.bike_info.getInfoFromCsv(conditions_path + "Phoenix_Matilde.csv")
run1.readRun(file_name)

run2 = Run()
file_name = dataset_path + "Diego_15_09_2023_AM_2.csv"
run2.bike_info.getInfoFromExcel(conditions_path + "Cerberus_Diego.xlsx")
run2.readRun(file_name)
run2.gearChangeDetect(initial_gear=1)
an_run.addRun(run2)

run3 = Run()
file_name = dataset_path + "Diego_13_09_2023_AM_2.csv"
run3.bike_info.getInfoFromCsv(conditions_path + "Cerberus_Diego.csv")
run3.readRun(file_name)
run2.gearChangeDetect(initial_gear=1)
an_run.addRun(run3)

an_run.comparation(opts="Diego", export_PDF=True, show=True)   #comparation between specified or default run with complete arbitrariness on the management of graphs

#___________________
## Run object : manual initialization
#
# for Matilde
# gb1 = GearBox(gear_box=[40,35,31,27,24,21,19,17,15,14,13,12], chainring=108, sec_ratio=[38,18])
# wl1 = Wheels(tyre="Michelin-blue", radius=0.23157)
# bk1 = BikeInfo(Vehicle("Phoenix"), Driver("Matilde"), wl1, gb1)
# run1.setBikeInfo(bk1)
# for Diego:
# gb3 = GearBox(gear_box=[32,28,24,21,19,17,15], chainring=60, sec_ratio=[54,17])
# wl3 = Wheels("Michelin-blue",None,0.23157,None,None)
# bk3 = BikeInfo(Vehicle("Cerberus"),Driver("Diego"),wl3,gb3)
# run3.setBikeInfo(bk3)

#___________________
## other stuff we can do with a Run object
#
# run2.setBounds(lwbd=3,upbd=3)  #cut 3 lines from the beginning and the end of the dataset
# run2.gearChangeDetect()
# run2.exportCols("prova.csv", ['gear', 'speed'])   #export a csv file with selected cols
# run2.plot(cols=["speed","ideal_speed"])
# run2.export()   #export a PDF file with the principal graphs
# print(run2.disp)   #displacement

#___________________
## RunAnalysis object
#
# an_run.plotEach(export=True)   #plot and export each run, representing specified or default cols
# an_run.comparation(keys=["Diego_15_09_2023_AM_2.csv","Diego_13_09_2023_AM_2.csv"],
#                    cols=[["speed","heart_rate"],["power", "heart_rate"],["ideal_speed","power","altitude"]],export_PDF=True)   #complete arbitrariness on the management of graphs
# an_run.calcAvgRun()
# an_run.run_list["avg_run"].plot()   #plot the average run

# model and simulation
# an_run.modeling(degree=3, input_values=["power","heart_rate","cadence"],output_value="speed",plot=True)   #create the model based on power, heart rate and cadence
# an_run.simulate(plot=True)














'''
# do not consider

# print(run1.run_data.iloc[0:7][['speed','gear','RPMw_bo_RPMp','ideal_speed']])
# print(run1.run_data[['speed','gear','cadence','RPMw_bo_RPMp','ideal_speed']])

# print(run2.bike_info.bike.chassis_weight)
# print(math.isnan(run2.bike_info.bike.chassis_weight))

# print(run3.bike_info.bike.chassis_weight)
# print(math.isnan(run3.bike_info.bike.chassis_weight))
'''
