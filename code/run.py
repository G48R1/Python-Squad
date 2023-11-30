from conditions import *
import numpy as np
import matplotlib.pyplot as plt
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
        self.header = None   #header of csv file
        self.n_data = None   #length of run_data
        self.disp = None   #displacement / dissipation factor

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
        self.n_data = len(self.run_data)
        #self.run_data["torq"] =self.run_data["Power"]*60/self.data["Cadence"] # wheel or pedal rpm? (here pedal)
        #self.run_data["wheel rpm"]=self.run_data["Cadence"]
        
    def gearChangeDetect(self, initial_gear=1):
        '''initial_gear: Integer'''
        #initializing variables
        rd = self.run_data
        bi = self.bike_info
        L = self.n_data
        rd['gear'] = np.ones(L)
        rd['gear'] = pd.to_numeric(rd['gear'],downcast='integer')
        rd.at[0,'gear'] = initial_gear
        rd['RPMw_bo_RPMp'] = np.ones(L)   #RPM wheel based on RPM pedal
        self.run_data['RPMw_bo_RPMp'] = pd.to_numeric(rd['RPMw_bo_RPMp'])
        max_gear = len(bi.gear_box.gear_box)

        #set first value of 'gear' and 'RPMw_bo_RPMp'
        T = bi.gear_box.gear_box[rd.at[0,'gear']-1]
        rd.at[0,'RPMw_bo_RPMp'] = rd.at[0,'cadence']*(bi.gear_box.chainring/T)*(bi.gear_box.sec_ratio[0]/bi.gear_box.sec_ratio[1])

        #calculating values of 'gear' and 'RPMw_bo_RPMp' (from 1 to n_data-3)
        for i in np.arange(self.n_data-3)+1:
            coeff1 = 0.95 #+ 0.02*(rd.at[i-1,'gear']/max_gear)**2   #variable depending on the gear (>gear  -->  >coeff)
            coeff2 = 0.98
            if rd.at[i,'cadence'] < rd.at[i-1,'cadence']*coeff1 and rd.at[i+2,'cadence']>=rd.at[i+1,'cadence']*coeff2:
                rd.at[i,'gear'] = rd.at[i-1,'gear'] + 1
            else:
                rd.at[i,'gear'] = rd.at[i-1,'gear']
            T = bi.gear_box.gear_box[rd.at[i,'gear']-1]
            rd.at[i,'RPMw_bo_RPMp'] = rd.at[i,'cadence']*(bi.gear_box.chainring/T)*(bi.gear_box.sec_ratio[0]/bi.gear_box.sec_ratio[1])

        #set last 2 values of 'gear' and 'RPMw_bo_RPMp'
        rd.at[self.n_data-2,'gear'] = rd.at[self.n_data-3,'gear']
        T = bi.gear_box.gear_box[rd.at[self.n_data-2,'gear']-1]   #theet of the sprockets / denti dei pignoni
        rd.at[self.n_data-2,'RPMw_bo_RPMp'] = rd.at[self.n_data-2,'cadence']*(bi.gear_box.chainring/T)*(bi.gear_box.sec_ratio[0]/bi.gear_box.sec_ratio[1])
        rd.at[self.n_data-1,'gear'] = rd.at[self.n_data-2,'gear']
        T = bi.gear_box.gear_box[rd.at[self.n_data-1,'gear']-1]   #theet of the sprockets / denti dei pignoni
        rd.at[self.n_data-1,'RPMw_bo_RPMp'] = rd.at[self.n_data-1,'cadence']*(bi.gear_box.chainring/T)*(bi.gear_box.sec_ratio[0]/bi.gear_box.sec_ratio[1])

        #calculating 'ideal_speed'
        rd['ideal_speed'] = rd['RPMw_bo_RPMp']*bi.wheels.radius*(np.pi/30)*3.6
        
        #calculating the dissipation factor
        self.disp = np.mean(abs(rd['speed'] - rd['ideal_speed']) / rd['speed'])
    
    def setBounds(self, lwbd = 2, upbd = None, all = False):   #actually the option 'all' it's not necessary
        '''set upper and lower limits of data (Trust-based) / limiti basati sulla attendibilità
        starting from 0 to n_data-1
        lwbd: Iterator (uInt)
        upbd: Iterator (uInt)
        all: Bool'''
        if all == True:
            lwbd = 0
            upbd = self.n_data
        else:
            if upbd == None:
                upbd = self.n_data - 2
            lwbd = max(lwbd,0)
            upbd = min(upbd,self.n_data)
        
        #changing data directly
        data = self.run_data.iloc[lwbd:upbd].values   #selecting the new bounded dataset
        names = self.run_data.columns.values   #getting the names of the columns
        self.run_data = pd.DataFrame(data,columns=names)
        self.n_data = len(self.run_data)
        #return self.run_data.iloc[lwbd:upbd]

    def exportCols(self, file_name, cols, rows = None):
        '''
        file_name: String
        rows: list of Iterator (uInt) [Default: all]
        cols: List of index (String/column name)
        '''
        if rows == None:
            rows = np.arange(self.n_data)
        util.writeFile(file_name, self.run_data.iloc[rows][cols].values, cols)

class RunAnalysis:
    
    def __init__(self):
        self.run_list = {}   #dictionary
    
    def addRun(self, run):
        '''run: Run Object'''
        if run.id_run not in self.run_list:
            self.run_list[run.id_run] = run
        else:
            print("run "+run.id_run+" already uploaded")
        
    def rmRun(self, id_run):
        '''id_run: String'''
        self.run_list.pop(id_run,'Not found')

    def plotEach(self, export=False):
        '''export: Bool'''
        for run in self.run_list.values():
            plt.plot(run.run_data["distance"],run.run_data["speed"], label="GPS speed")
            plt.plot(run.run_data["distance"],run.run_data["ideal_speed"],label="ideal speed")
            plt.plot(run.run_data["distance"],run.run_data["power"],label="power")
            if "heart_rate" in run.run_data.columns.values:
                plt.plot(run.run_data["distance"],run.run_data["heart_rate"], label="heart_rate")
            plt.title("Data")
            plt.legend()
            plt.show()

