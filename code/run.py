from conditions import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
# from nfoursid.kalman import Kalman
# from nfoursid.nfoursid import NFourSID
# from nfoursid.state_space import StateSpace
from sklearn.preprocessing import PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import util
import os
os.chdir(os.path.dirname(__file__))   #set the right path (could help in vscode cause sometime it is dumb)

class Run:
    
    def __init__(self):
        '''
        id_run: String (file name)
        atm_cond: AtmConditions Object
        bike_info: BikeInfo Object
        run_data: DataFrame
        n_data: Int (number of run_data's rows)
        disp: Float (displacement)
        avg_values: Dict
        '''
        self.id_run = None
        self.atm_cond = None
        self.bike_info = BikeInfo()
        self.run_data = None
        self.n_data = None   #length of run_data
        self.disp = None   #displacement / dissipation factor
        self.avg_values = {}
        
    def clean(self):
        self.id_run = None
        self.atm_cond = None
        # self.bike_info = BikeInfo()
        self.run_data = None
        self.n_data = None
        self.disp = None
        self.avg_values = {}        

    def setAtmCond(self, atm_cond):
        '''
        atm_cond: AtmConditions Object
        set atmospheric conditions
        '''
        self.stm_cond = atm_cond
    
    def setBikeInfo(self, bike_info):
        '''
        bike_info: BikeInfo Object
        set bike info
        '''
        self.bike_info = bike_info

    def readRun(self, file_name, cut=True):
        self.id_run = file_name.rsplit('/',1)[-1].replace(".csv","")   #extraction of file name from path
        self.run_data = util.csv2Df(file_name)
        self.run_data['timestamp'] = pd.to_datetime(self.run_data['timestamp'])
        for index in self.indexes():
            if index != "timestamp":
                self.run_data[index] = pd.to_numeric(self.run_data[index])
        self.n_data = len(self.run_data)
        if cut==True:
            self.setBounds(lwbd=2,upbd=2)
        self.gearChangeDetect()   #comment this line if there isn't enough bike info

    def rescale(self, col="altitude", min_bd=50):
        '''
        col: String (Index)
        min_bd: Int ["min bound"]
        rescale values in col in the way the min value is min_bd
        '''
        min = np.min(self.run_data[col])
        return self.run_data[col] - min + min_bd
        
    def addCol(self, col_name, col):
        '''
        col_name: String
        col: List/DataFrame/Series of data
        add a column inside the dataset
        '''
        if len(col) == self.n_data:
            self.run_data[col_name] = col
            self.run_data[col_name] = pd.to_numeric(self.run_data[col_name])
        else:
            print("length not equal: ",len(col)," not equal to ",self.n_data)
        self.calcAvgValues()
    
    def indexes(self):
        return self.run_data.columns.values
    
    def calcAvgValues(self):
        '''
        calculate average values of each column except for "timestamp" and "distance"
        '''
        cols = self.indexes()
        cols = np.delete(cols, np.where(cols == "timestamp"))
        cols = np.delete(cols, np.where(cols == "distance"))
        for index in cols:
            self.avg_values[index] = np.mean(self.run_data[index])
        for index in cols:
            self.avg_values["std_"+index] = np.std(self.run_data[index])
    
    def gearChangeDetect(self, initial_gear=1):
        '''
        initial_gear: Integer
        detect the gear change and calculate the ideal speed
        necessary:
        run_data [cadence, speed]
        bike_info
        {
            bike [None]
            driver [None]
            wheels [radius]
            gear_box [gear_box (list), chainring, sec_ratio]
        }
        '''
        #initializing variables
        rd = self.run_data
        bi = self.bike_info
        L = self.n_data
        rd['gear'] = np.ones(L)*initial_gear
        rd['gear'] = pd.to_numeric(rd['gear'],downcast='integer')
        # rd.at[0,'gear'] = initial_gear
        rd['RPMw_bo_RPMp'] = np.ones(L)   #RPM wheel based on RPM pedal
        self.run_data['RPMw_bo_RPMp'] = pd.to_numeric(rd['RPMw_bo_RPMp'])
        max_gear = len(bi.gear_box.gear_box)

        #set first value of 'gear' and 'RPMw_bo_RPMp'
        T = bi.gear_box.gear_box[rd.at[0,'gear']-1]
        rd.at[0,'RPMw_bo_RPMp'] = rd.at[0,'cadence']*(bi.gear_box.chainring/T)*(bi.gear_box.sec_ratio[0]/bi.gear_box.sec_ratio[1])

        #calculating values of 'gear' and 'RPMw_bo_RPMp' (from 1 to n_data-3)
        for i in np.arange(self.n_data-3)+1:
            coeff1 = 0.95 #+ 0.02*(rd.at[i-1,'gear']/max_gear)**2   #variable depending on the gear (>gear  -->  >coeff)
            coeff2 = 1 - 0.02
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
        self.calcAvgValues()
    
    def setBounds(self, lwbd = 2, upbd = None, all = False):   #actually the option 'all' it's not necessary
        '''
        lwbd: Int
        upbd: Int
        all: Bool
        set upper and lower limits of data (Trust-based) / limiti basati sulla attendibilità
        starting from 0 to n_data-1
        '''
        if all == True:
            lwbd = 0
            upbd = 0
        else:
            if not bool(upbd):
                upbd = 2
            lwbd = max(lwbd,0)
            upbd = max(upbd,0)
        upbd = self.n_data - upbd        
        #changing data directly
        data = self.run_data.iloc[lwbd:upbd].values   #selecting the new bounded dataset
        names = self.indexes()   #getting the names of the columns
        self.run_data = pd.DataFrame(data,columns=names)
        self.n_data = len(self.run_data)
        #return self.run_data.iloc[lwbd:upbd]
        self.calcAvgValues()
        self.gearChangeDetect()

    def exportCols(self, file_name, cols, rows = None):
        '''
        file_name: String
        rows: list of Iterator (uInt) [Default: all]
        cols: List of index (String/column name)
        export some cols in a csv file
        '''
        if rows == None:
            rows = np.arange(self.n_data)
        util.writeCsvFile(file_name, self.run_data.iloc[rows][cols].values, cols)
    
    def plot(self, cols=[], alt_min_bd=50, export=False, show=True):
        '''
        cols: List of index (String/column name)  default: ["speed", "ideal_speed", "power", "heart_rate"]
        alt_min_bd: int [min bound of rescaled altitude]
        plot the graphs of specific or default cols
        '''
        if cols == []:
            cols = ["speed", "ideal_speed", "power", "heart_rate"]
        for col in cols:
            if col in self.indexes():
                if col=='altitude':
                    plt.plot(self.run_data["distance"],self.rescale(col,alt_min_bd),label=col)
                    h_i = self.run_data.at[0,"altitude"]
                    h_f = self.run_data.at[self.n_data-1,"altitude"]
                    # h_min = min(self.run_data["altitude"])
                    # h_max = max(self.run_data["altitude"])
                    marginex1 = - 200/8000 * (self.run_data.at[self.n_data-1,"distance"]-self.run_data.at[0,"distance"])
                    marginey1 = -4
                    marginex2 = + 200/8000 * (self.run_data.at[self.n_data-1,"distance"]-self.run_data.at[0,"distance"])
                    marginey2 = -3.5
                    plt.text(self.run_data.at[0,"distance"] + marginex1, self.rescale(col,alt_min_bd)[0] + marginey1, "h_i : " + str("%.2f" % h_i) + "m")
                    plt.text(self.run_data.at[self.n_data-1,"distance"] + marginex2, self.rescale(col,alt_min_bd)[self.n_data-1] + marginey2, "h_f : " + str("%.2f" % h_f) + "m", horizontalalignment="right")
                else:
                    plt.plot(self.run_data["distance"],self.run_data[col],label=col)
        plt.title("Data: run "+self.id_run)
        plt.legend()
        if export==True:
            plt.savefig(self.id_run+".pdf")
        if show==True:
            plt.show()
    
    def export(self):   #useless
        '''
        export PDF with graphs of principal cols
        '''
        # Create an PdfPages object to save multiple plots in a single PDF
        with PdfPages(self.id_run+'.pdf') as pdf:
            plt.plot(self.run_data["distance"],self.run_data["speed"],label="GPS speed")
            if "ideal_speed" in self.indexes():
                plt.plot(self.run_data["distance"],self.run_data["ideal_speed"],label="ideal speed")
            plt.plot(self.run_data["distance"],self.run_data["power"],label="power")
            max_power = max(self.run_data["power"])*np.ones(self.n_data)
            plt.plot(self.run_data["distance"],max_power,label="power max",alpha=0.5)
            plt.title("Data: run "+self.id_run)
            plt.legend()
            pdf.savefig(bbox_inches='tight', pad_inches=0.5)
            plt.close()
            if "heart_rate" in self.indexes():
                plt.plot(self.run_data["distance"],self.run_data["heart_rate"],label="heart rate")
            plt.title("Data: run "+self.id_run)
            plt.legend()
            # Save plot in the PDF file
            pdf.savefig(bbox_inches='tight', pad_inches=0.5)
            plt.close()
    
    def generateCol(self, col="power", avg_value=None, std_value=None, std_perc=None):
        '''
        col: String (Index) default: power
        avg_value: Float
        std_value: Float
        std_perc: Float
        generate a random column with specified mean, standard deviation and/or percentage standard deviation
        '''
        if not bool(avg_value):
            avg_value = self.avg_values[col]
        if not bool(std_value):
            std_value = self.avg_values["std_"+col]
        if not bool(std_perc):
            std_perc = std_value/avg_value
        scatter_factor = (1-std_perc) + (2*std_perc)*np.random.random(self.n_data)
        return scatter_factor*avg_value

class RunAnalysis:
    
    def __init__(self):
        self.run_list = {}   #dictionary
        self.poly = None
        self.model = None
        self.model_data = None
        self.prediction = None
        self.dict_opts = {
            "def": [["speed","power","ideal_speed"],["altitude","heart_rate"]],
            "Diego": [["speed","power"],["altitude","heart_rate"],["speed","ideal_speed"]],
            "Matilde": [["speed","power"],["heart_rate"]],
            "Enzo": [["speed","power"],["speed","ideal_speed"]]
        }
    
    def addRun(self, run):
        '''
        run: Run Object
        add a Run object to the dictionary
        '''
        if run.id_run not in self.run_list:
            self.run_list[run.id_run] = run
        else:
            print("run "+run.id_run+" already uploaded")
        
    def rmRun(self, id_run):
        '''
        id_run: String
        remove a Run object from the dictionary
        '''
        self.run_list.pop(id_run,'Not found')

    def uploadFolder(self, folder_path, conds_file):
        '''
        folder_path : String (Path)
        conds_file : String (Excel file)
        upload all races that are in a folder
        '''
        # run = Run()
        # run.bike_info.getInfoFromExcel(conds_file)
                
        for file in os.listdir(folder_path):
            if ".csv" in file:
                run = Run()
                run.bike_info.getInfoFromExcel(conds_file)
                # run.clean()
                run_path = os.path.join(folder_path, file)
                run_path = run_path.replace("\\","/")
                run.readRun(run_path)
                self.addRun(run)

    def plotEach(self, cols=[], export=False):
        '''
        cols: List of index (String/column name)  default: ["speed", "ideal_speed", "power", "heart_rate"]
        export: Bool
        plot one graph with specified or default cols for each run
        '''
        if cols == []:
            cols = ["speed", "ideal_speed", "power", "heart_rate"]
        for run in self.run_list.values():
            run.plot(cols=cols, export=export)
            # for col in cols:
            #     if col in run.indexes():
            #         plt.plot(run.run_data["distance"],run.run_data[col],label=col)
            # plt.title("Data: run "+str(i+1))
            # plt.legend()
            # plt.show()

    def comparation(self, keys=None, cols=[], opts="def", export_PDF=False, export_PNG=False, show=True):
        '''
        keys: List of String (run ID)  default: all
        cols: List of List of Index (String/column name)  default: opts
        opts: "def", "Diego", "Matilde", "Enzo"
        allow to comparate specified in cols of two or more races (listed in keys)
        '''
        if cols==[]:
            cols = self.dict_opts[opts]
        if not bool(keys):
            keys = self.run_list.keys()
        if export_PDF==True:
            pdf = PdfPages("comparation"+".pdf")
        for plot in cols:
            flag = False
            for i, key in enumerate(keys):
                run = self.run_list.get(key)
                id = " run "+ str(i+1)
                for col in plot:
                    if col in run.indexes():
                        flag = True
                        if col=="altitude":
                            plt.plot(run.run_data["distance"],run.rescale(col),label=col+id)
                            # h_i = run.run_data.at[0,"altitude"]
                            # h_f = run.run_data.at[run.n_data-1,"altitude"]
                            # # h_min = min(run.run_data["altitude"])
                            # # h_max = max(run.run_data["altitude"])
                            # marginex1 = - 200/8000 * (run.run_data.at[run.n_data-1,"distance"]-run.run_data.at[0,"distance"])
                            # marginey1 = -4
                            # marginex2 = + 200/8000 * (run.run_data.at[run.n_data-1,"distance"]-run.run_data.at[0,"distance"])
                            # marginey2 = -3.5
                            # plt.text(run.run_data.at[0,"distance"] + marginex1, run.rescale(col)[0] + marginey1, "h_i : " + str("%.2f" % h_i) + "m")
                            # plt.text(run.run_data.at[run.n_data-1,"distance"] + marginex2, run.rescale(col)[run.n_data-1] + marginey2, "h_f : " + str("%.2f" % h_f) + "m", horizontalalignment="right")
                        else:
                            plt.plot(run.run_data["distance"],run.run_data[col],label=col+id)
            if flag==True:
                plt.title("Comparation")
                plt.legend()
                if export_PDF==True:
                    pdf.savefig(bbox_inches='tight', pad_inches=0.5)
                if export_PNG==True:
                    fname = plot[0]
                    for col in plot[1:]:
                        fname = fname + "_" + col
                    plt.savefig(fname=fname+".png")
                if show==True:
                    plt.show()
                plt.close()
        if export_PDF==True:
            pdf.close()

    # def export(self):
    #     # Create an PdfPages object to save multiple plots in a single PDF
    #     with PdfPages(self.id_run+'.pdf') as pdf:
    #         plt.plot(self.run_data["distance"],self.run_data["speed"],label="GPS speed")
    #         if "ideal_speed" in self.indexes():
    #             plt.plot(self.run_data["distance"],self.run_data["ideal_speed"],label="ideal speed")
    #         plt.plot(self.run_data["distance"],self.run_data["power"],label="power")
    #         max_power = max(self.run_data["power"])*np.ones(self.n_data)
    #         plt.plot(self.run_data["distance"],max_power,label="power max",alpha=0.5)
    #         plt.title("Data: run "+self.id_run)
    #         plt.legend()
    #         pdf.savefig(bbox_inches='tight', pad_inches=0.5)
    #         plt.close()
    #         if "heart_rate" in self.indexes():
    #             plt.plot(self.run_data["distance"],self.run_data["heart_rate"],label="heart rate")
    #         plt.title("Data: run "+self.id_run)
    #         plt.legend()
    #         # Save plot in the PDF file
    #         pdf.savefig(bbox_inches='tight', pad_inches=0.5)
    #         plt.close()

        
    def calcAvgRun(self):   #da controllare se sono di lunghezze diverse
        '''
        calculate average run
        '''
        if not self.run_list:
            print("no run in run_list")
            return
        n_data = float('inf')
        for run in self.run_list.values():
            n_data = min(n_data, run.n_data)
        for run in self.run_list.values():   #tolgo i dati all'inizio (assumo che i dati alla fine siano sincronizzati)
            run.setBounds(lwbd=run.n_data-n_data)
        rlv = list(self.run_list.values())
        count_index = {}
        avg_run = Run()
        avg_run.id_run = "avg_run"
        avg_run.run_data = rlv[0].run_data
        avg_run.n_data = n_data
        cols = avg_run.indexes()
        cols = np.delete(cols, np.where(cols == "timestamp"))
        cols = np.delete(cols, np.where(cols == "distance"))
        for col in cols:
            count_index[col] = 1
        for run in rlv[1:]:
            run_cols = run.indexes()
            run_cols = np.delete(run_cols, np.where(run_cols == "timestamp"))
            run_cols = np.delete(run_cols, np.where(run_cols == "distance"))

            for col in run_cols:
                if col not in cols:
                    avg_run.run_data[col] = run.run_data[col]
                    count_index[col] = 1
                else:
                    avg_run.run_data[col] = avg_run.run_data[col] + run.run_data[col]
                    count_index[col] = count_index[col]+1
        for col in cols:
            avg_run.run_data[col] = avg_run.run_data[col]/count_index[col]
        
        avg_run.calcAvgValues()
        self.addRun(avg_run)

    def generateCol(self, col="power", avg_value=None, std_value=None, std_perc=None):
        '''
        col: String (Index) default: power
        avg_value: Float
        std_value: Float
        std_perc: Float
        generate a random column with specified mean, standard deviation and/or percentage standard deviation
        '''
        if "avg_run" not in self.run_list.keys():
            self.calcAvgRun()
        if not bool(avg_value):
            avg_value = self.run_list.get("avg_run").avg_values[col]
        if not bool(std_value):
            std_value = self.run_list.get("avg_run").avg_values["std_"+col]
        if not bool(std_perc):
            std_perc = std_value/avg_value
        scatter_factor = (1-std_perc) + (2*std_perc)*np.random.random(self.run_list["avg_run"].n_data)   #(self.run_list["Diego_13_09_2023_AM_2.csv"].n_data)
        return scatter_factor*avg_value

    def modeling(self, degree=2, input_values=['power', 'heart_rate'], output_value="speed", plot=False):
        '''
        degree: Int
        input_values: List of String (Index)
        output_value: String (Index)
        plot: Bool
        create a model to predict output_value knowing the input_values.
        model used: Polynomial Regression of degree=degree
        '''
        all_values = []
        all_values.extend(input_values)
        all_values.append(output_value)
        self.model_data = input_values
        self.prediction = output_value
        for run in self.run_list.values():
            for value in all_values:
                if value not in run.indexes():
                    print("no data for \""+value+"\"")
                    return
        
        self.poly = PolynomialFeatures(degree)
        self.model = LinearRegression()

        datasets = self.run_list.values()
        # Addestra e valuta il modello su ciascun dataset
        for i, dataset in enumerate(datasets):
            X = dataset.run_data[input_values].values
            y = dataset.run_data[output_value].values
            
            # per evitare il warning (NON TESTATO)
            # preprocessor = ColumnTransformer(transformers=[('poly', self.poly, dataset.columns)],remainder='passthrough')
            # X_poly = preprocessor.fit_transform(dataset)

            X_poly = self.poly.fit_transform(X)
            self.model.fit(X_poly, y)

            predictions = self.model.predict(X_poly)

            mse = mean_squared_error(y, predictions)
            r2 = r2_score(y, predictions)

            print(f'Run {i + 1} - MSE: {mse}, R²: {r2}')
            
            if plot==True:
                # Visualize the results for each dataset
                plt.scatter(X[:, 0], y, label=f'Dataset {i + 1} - Real Data', alpha=0.5)
                plt.scatter(X[:, 0], predictions, label=f'Dataset {i + 1} - Predictions (Array)', linewidth=2)
        if plot==True:
            plt.xlabel(input_values[0])
            plt.ylabel(output_value+' Profile')
            tt = ""
            for index in input_values:
                tt = tt+" "+index
            plt.title(f'Polynomial Regression (Degree {degree}) with'+tt)
            plt.legend()
            plt.show()
    
    def simulate(self, input_values=None, plot=False, export=False):
        '''
        input_values: DataFrame
        plot: Bool
        export: Bool
        simulate self.prediction knowing input_values, using self.model
        '''
        if self.model==None:
            print("No model")
            return
        if input_values==None:
            tmp = {}
            for index in self.model_data:
                tmp[index] = self.generateCol(col=index)
            input_values = pd.DataFrame(tmp)
            
        if not input_values.columns.equals(pd.Index(self.model_data)):
            print("Columns must match")
            return

        simulated_data = input_values
        X_sim_poly = self.poly.transform(simulated_data)
        simulated_data[self.prediction] = self.model.predict(X_sim_poly)
        
        sim_run = Run()
        sim_run.id_run = "sim_run"
        sim_run.n_data = self.run_list["avg_run"].n_data
        sim_run.run_data = simulated_data
        sim_run.run_data["distance"] = self.run_list["avg_run"].run_data["distance"]
        sim_run.calcAvgValues()
        self.addRun(sim_run)
        
        if plot==True:
            tmp = np.arange(0,len(simulated_data[self.prediction]))
            plt.scatter(tmp, simulated_data[self.prediction], color='red', label='Simulated '+self.prediction)
            for index in self.model_data:
                plt.scatter(tmp, simulated_data[index], color='orange', label=index)
            plt.xlabel('distance')
            plt.ylabel(self.prediction+' Profile')
            plt.title('Simulation Results')
            plt.legend()
            plt.show()
            
            # plt.scatter(simulated_data['distance'], simulated_data[self.prediction], color='red', label='Simulated '+self.prediction)
            # for index in self.model_data:
            #     plt.scatter(simulated_data['distance'], simulated_data[index], color='orange', label=index)
            # plt.xlabel('distance')
            # plt.ylabel(self.prediction+' Profile')
            # plt.title('Simulation Results')
            # plt.legend()
            # plt.show()
            
        if export==True:
            pass

    
