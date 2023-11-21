import conditions as cond
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
        data, self.header = util.readFile(file_name) 
        self.run_data = pd.DataFrame(data, columns = self.header)
        #self.run_data["speed"] = self.data["speed"]/3.6   #conversion from km/h to m/s
        n = len(self.run_data)
        #self.run_data["time"]
        #self.run_data["torq"] =self.run_data["Power"]*60/self.data["Cadence"] # wheel or pedal rpm? (here pedal)
        #self.run_data["wheel rpm"]=self.run_data["Cadence"]


run1 = Run()
run1.readRun("../Dataset/Matilde_13_09_2023_AM.csv")
print(run1.run_data['speed'])
