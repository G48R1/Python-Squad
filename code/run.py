import conditions as cond
import numpy as np
import pandas as pd
import util
import os
os.chdir(os.path.dirname(__file__))   #set the right path (could help in vscode cause sometime it is dumb)

class Run:
    
    def __init__(self):
        '''
        atm_cond: AtmConditions Object
        bike_info: BikeInfo Object
        run_data: List (Matrix)
        header: List of String (header of csv file)
        '''
        self.id_run = None
        self.atm_cond = None
        self.bike_info = None
        self.run_data = None
        self.header = None

    def setAtmCond(self, atm_cond):
        '''atm_cond: AtmConditions Object'''
        self.stm_cond = atm_cond
    
    def setBikeInfo(self, bike_info):
        '''bike_info: BikeInfo Object'''
        self.bike_info = bike_info

    def readRun(self, file_name):
        '''file_name: String (Path)'''
        self.id_run = file_name.rsplit('/',1)[-1]   #extraction of file name from path
        data, self.header = util.readFile(file_name)
        self.run_data = pd.DataFrame(data, columns = self.header)
        self.run_data['cadence'] = pd.to_numeric(self.run_data['cadence'])
        self.run_data['speed'] = pd.to_numeric(self.run_data['speed'])
        self.run_data['power'] = pd.to_numeric(self.run_data['power'])
        # self.run_data['distance'] = pd.to_numeric(self.run_data['distance'])
        # self.run_data['altitude'] = pd.to_numeric(self.run_data['altitude'])
        
        #self.run_data["speed"] = self.data["speed"]/3.6   #conversion from km/h to m/s
        #n = len(self.run_data)
        #self.run_data["torq"] =self.run_data["Power"]*60/self.data["Cadence"] # wheel or pedal rpm? (here pedal)
        #self.run_data["wheel rpm"]=self.run_data["Cadence"]
        
    def gearChangeDetect(self, initial_gear=1):
        '''initial_gear: Integer'''
        L = len(self.run_data['cadence'])
        self.run_data['gear'] = np.zeros(L)
        self.run_data['gear'] = pd.to_numeric(self.run_data['gear'],downcast='integer')
        self.run_data.at[0,'gear'] = initial_gear

        RPM_wheel_to_RPM_pedal = np.zeros(L)
        max_gear = len(self.bike_info.gear_box.gear_box)

        for i in np.arange(len(self.run_data['cadence'])-2)+1:
            coeff = 0.95 #+ 0.02*(self.run_data.at[i-1,'gear']/max_gear)**2   #variable depending on the gear (>gear  -->  >coeff)
            if self.run_data.at[i,'cadence'] < self.run_data.at[i-1,'cadence']*coeff and self.run_data.at[i+1,'cadence']>=self.run_data.at[i,'cadence']*0.98:
                self.run_data.at[i,'gear'] = self.run_data.at[i-1,'gear'] + 1
            else:
                self.run_data.at[i,'gear'] = self.run_data.at[i-1,'gear']
            T = self.bike_info.gear_box.gear_box[self.run_data.at[i,'gear']]
            RPM_wheel_to_RPM_pedal[i] = self.run_data.at[i,'cadence']*self.bike_info.gear_box.chainring/T*self.bike_info.gear_box.sec_ratio[0]/self.bike_info.gear_box.sec_ratio[1]

        self.run_data.at[len(self.run_data['cadence'])-1,'gear'] = self.run_data.at[len(self.run_data['cadence'])-2,'gear']
        T = self.bike_info.gear_box.gear_box[self.run_data.at[len(self.run_data['cadence'])-1,'gear']]
        RPM_wheel_to_RPM_pedal[-1] = self.run_data.at[len(self.run_data['cadence'])-1,'cadence']*self.bike_info.gear_box.chainring/T*self.bike_info.gear_box.sec_ratio[0]/self.bike_info.gear_box.sec_ratio[1]

        self.run_data['ideal_speed'] = RPM_wheel_to_RPM_pedal*self.bike_info.wheels.radius*(np.pi/30)*3.6

class RunAnalysis:
    
    def __init__(self):
        self.run_list = []
        self.runs_id = []
    
    def addRun(self, run):
        '''run: Run Object'''
        if run.id_run not in self.runs_id:
            self.run_list.append(run)
            self.runs_id.append(run.id_run)
        
    def rmRun(self, id_run):
        '''id_run: String'''
        for i in len(self.run_list):
            if self.run_list[i].id_run == id_run:
                self.run_list.pop(i)
                self.runs_id.pop(i)

    def plotEach(self, export):
        '''export: Bool'''
        for run in self.run_list:
            pass            


## main di prova ##

run1 = Run()
file_name = "../Dataset/Matilde_13_09_2023_AM.csv"
run1.readRun(file_name)
gb1 = cond.GearBox([40,35,31,27,24,21,19,17,15,14,13,12])
gb1.setInfo(chainring=108, sec_ratio=[38,18])
wl1 = cond.Wheel("Michelin-blue")
wl1.setInfo(None,0.23157,None,None)
bk1 = cond.BikeInfo(cond.Vehicle("Phoenix"),cond.Driver("Matilde"),wl1,gb1)
run1.setBikeInfo(bk1)
#print(run1.run_data['speed'])
# print(run1.id_run)
run1.gearChangeDetect()
print(run1.run_data['gear'])
#print(run1.run_data['ideal_speed'])
#print(run1.run_data['speed'])

run2 = Run()
file_name = "../Dataset/Diego_15_09_2023_AM_2.csv"
run2.readRun(file_name)
#print(run2.run_data['speed'])
print(run2.id_run)

an_run = RunAnalysis()
an_run.addRun(run1)
an_run.addRun(run2)
print(an_run.runs_id)

