import conditions as cond
import pandas as pd
import csv
import util
import os
os.chdir(os.path.dirname(__file__))

class Run:
    
    def __init__(self):
        self.atm_cond = None
        self.bike_info = None
        self.run_data = None
        self.header = None

    def setAtmCond(self, atm_cond):
        self.stm_cond = atm_cond
    
    def setBikeInfo(self, bike_info):
        self.bike_info = bike_info

    def readRun(self, file_name):
        data, self.header = util.readFile(file_name) 
        self.run_data = pd.DataFrame(data, columns = self.header)
        #self.run_data["speed"] = self.data["speed"]/3.6   #conversion from km/h to m/s
        n = len(self.run_data)
        #self.run_data["time"]
        #self.run_data["torq"] =self.run_data["Power"]*60/self.data["Cadence"] # wheel or pedal rpm? (here pedal)
        #self.run_data["wheel rpm"]=self.run_data["Cadence"] # TODO, they will give it to us


run1= Run()
run1.readRun("Matilde_13_09_2023_AM.csv")
print(run1.run_data['speed'])
print('ciao')