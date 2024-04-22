from conditions import *
import numpy as np
import matplotlib.pyplot as plt
import copy
from matplotlib import cm
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
os.chdir(os.path.dirname(__file__))   #set the right path (could help in vscode cause sometimes it is dumb)

class Run:
    
    def __init__(self, file_name=None, cond_file=None, settings_file=None):
        '''
        id_run: String (file name)
        atm_cond: AtmConditions Object
        bike_info: BikeInfo Object
        run_data: DataFrame
        n_data: Int (number of run_data's rows)
        disp: Float (displacement)
        avg_values: Dict
        '''
        self.num_id = None
        self.id_run = None
        self.atm_cond = None
        self.bike_info = BikeInfo()
        self.run_data = None
        self.n_data = None   #length of run_data
        self.disp = None   #displacement / dissipation factor
        self.avg_values = {}
        if file_name is not None:
            if settings_file is not None:
                self.readRun(file_name=file_name, settings_file=settings_file)
            else:
                self.setBikeInfo(cond_file=cond_file)
                self.readRun(file_name=file_name)
        
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
    
    def setSettings(self, file_name, settings_file):
        column_names = ["num", "idrun", "cond_file"]
        df = pd.read_excel(settings_file, header=1, names=column_names)
        # Delete void rows (NaN)
        df = df.dropna(axis=0, how='all')
        index = df.index[df.idrun == file_name+".csv"].values[0]
        self.num_id = df.at[index,"num"]
        cond_file = df.at[index,"cond_file"]
        cond_file = util.getCondPath(cond_file)
        self.setBikeInfo(cond_file=cond_file)

    def setBikeInfo(self, bike_info=BikeInfo(), cond_file=None):
        '''
        bike_info: BikeInfo Object
        set bike info
        '''
        if cond_file is not None:
            self.bike_info.getInfoFromExcel(cond_file)
        else:
            self.bike_info = bike_info

    def readRun(self, file_name, cut=True, gear_detect=True, settings_file=None):
        '''
        file_name : String (Path)
        cut : Bool (call setBounds)
        gear_detect : Bool (call gearChangeDetect)
        read data from csv file. rescale distance starting by 0
        '''
        self.id_run = file_name.rsplit('/',1)[-1].replace(".csv","")   #extraction of file name from path
        #######
        if settings_file is not None:
            self.setSettings(self.id_run, settings_file)
        #######
        self.run_data = util.csv2Df(file_name)
        self.setColsType()
        self.run_data["distance"] = self.rescale(col="distance",min_bd=0)
        self.n_data = len(self.run_data)
        if cut==True:
            self.setBounds(lwbd=2,upbd=2)
        if gear_detect==True:
            self.gearChangeDetect()   #comment this line if there isn't enough bike info

    def setColsType(self, cols=[]):
        '''
        cols : List of String (Index)
        '''
        if cols==[]:
            cols = self.indexes()
        else:
            cols = list(set(cols).intersection(self.indexes()))                
        for index in cols:
            if index == "timestamp":
                self.run_data['timestamp'] = pd.to_datetime(self.run_data['timestamp'])
            else:
                self.run_data[index] = pd.to_numeric(self.run_data[index])

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
            self.setColsType(cols=[col_name])
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
        self.setColsType(cols=["RPMw_bo_RPMp","ideal_speed"])
        rd['gear'] = pd.to_numeric(rd['gear'],downcast='integer')
        self.calcAvgValues()
    
    def calcDisplacement(self):
        self.disp = np.mean(abs(self.run_data['speed'] - self.run_data['ideal_speed']) / self.run_data['speed'])
    
    def setBounds(self, lwbd=2, upbd=2, all=False):   #actually the option 'all' it's not necessary
        '''
        lwbd: Int
        upbd: Int
        all: Bool
        set lower (begin) and upper (end) limits of data (Trust-based) / limiti basati sulla attendibilità
        starting from 0 to n_data-1
        '''
        if all==True:
            lwbd = 0
            upbd = 0
        else:
            lwbd = max(lwbd,0)
            upbd = max(upbd,0)
        upbd = self.n_data - upbd        
        #changing data directly
        data = self.run_data.iloc[lwbd:upbd].values   #selecting the new bounded dataset
        names = self.indexes()   #getting the names of the columns
        self.run_data = pd.DataFrame(data,columns=names)
        self.n_data = len(self.run_data)
        #return self.run_data.iloc[lwbd:upbd]
        self.setColsType()
        self.calcAvgValues()
        self.gearChangeDetect()

    def exportCols(self, file_name, cols, rows=None):
        '''
        file_name: String
        rows: list of Iterator (uInt) [Default: all]
        cols: List of index (String/column name)
        export some cols in a csv file
        '''
        if rows is None:
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
        else:
            cols = list(set(cols).intersection(self.indexes()))                
        for col in cols:
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
            pdfexport_path = util.joinPath(util.pdfexport_path,self.id_run)
            plt.savefig(pdfexport_path+".pdf")
        if show==True:
            plt.show()
    
    def export(self):   #useless
        '''
        export PDF with graphs of principal cols
        '''
        pdfexport_path = util.joinPath(util.pdfexport_path,self.id_run)
        # Create an PdfPages object to save multiple plots in a single PDF
        with PdfPages(pdfexport_path+'.pdf') as pdf:
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
    
    def __init__(self, settings_file=None, plot_opts_file=util.getPlotOptsPath()):
        self.run_list = {}   #dictionary
        self.num_run = {}
        self.settings = settings_file
        self._poly = None
        self._model = None
        self._model_data = None
        self._prediction = None
        self._dict_opts = None
        self.addDictOpts(plot_opts_file)
        
    def addDictOpts(self, plot_opts_file):        
        self._dict_opts = pd.read_excel(plot_opts_file,index_col=0,header=None).T.to_dict()
        for key in self._dict_opts.keys():
            strtmp = list()
            for val in self._dict_opts[key].values():
                if val is not np.nan:
                    strtmp.append(list(val.replace(' ','').split(',')))
            self._dict_opts[key] = strtmp
    
    def addSettings(self, settings_file):
        self.settings = settings_file
    
    def addRun(self, run=None, file_name=None, cond_file=None, replace=False):
        '''
        run: Run Object
        add a Run object to the dictionary
        '''
        if file_name is not None:
            if self.settings is not None:
                run = Run(file_name=file_name,settings_file=self.settings)
            else:
                run = Run(file_name=file_name,cond_file=cond_file)
        if (replace==False and run.id_run not in self.run_list) or replace==True:
            self.run_list[run.id_run] = run
        else:
            print("run "+run.id_run+" already uploaded")
        # self.AvgRunRunTime
        # self.calcAvgRun()
        
    def rmRun(self, id_run):
        '''
        id_run: String
        remove a Run object from the dictionary
        '''
        self.run_list.pop(id_run,'Not found')
        # self.calcAvgRun()

    def uploadFolder(self, folder_path, settings_file=None, replace=False):
        '''
        folder_path : String (Path)
        settings_file : String (Excel file)
        upload all races that are in a folder
        '''
        if self.settings is None:
            self.settings = settings_file
        # run = Run()
        # run.bike_info.getInfoFromExcel(conds_file)
        for file in os.listdir(folder_path):
            if ".csv" in file:
                # run = Run()
                # run.bike_info.getInfoFromExcel(conds_file)
                # run.clean()
                run_path = os.path.join(folder_path, file)
                # print(run_path)
                run_path = run_path.replace("\\","/")
                # print(run_path)
                # run.readRun(file_name=run_path)
                # self.addRun(run,replace=replace)
                self.addRun(file_name=run_path)

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

    def comparation(self, keys=[], cols="default", export_PDF=False, export_PNG=False, show=True, vis_max=[], pdf_name=""):
        '''
        keys: List of String (run ID)  default: all
        cols: List of List of Index (String/column name)  default opts: "default", "Diego", "Matilde", "Enzo"
        vis_max : List of String (Index) [visualize max]
        allow to comparate specified in cols of two or more races (listed in keys)
        '''
        cmap = cm.get_cmap('nipy_spectral')   #choose colormap : 'gist_rainbow', 'jet', 'hsv'

        if export_PDF==True:
            tmp = ""
            if isinstance(cols,str):
                tmp = "_"+cols
            pdfexport_path = util.joinPath(util.pdfexport_path,"comparation"+tmp+".pdf")
            pdf = PdfPages(pdfexport_path)   #TODO add driver name
        if isinstance(cols,str):
            cols = self._dict_opts[cols]
        if keys==[]:
            keys = self.run_list.keys()
        
        if cols == []:
            return

        delta_max_colors = 0.075   #change this parameter to change color gradient (delta) [0.05,0.1]
        a = 0.05
        b = 0.95
        delta_run = (b-a)/len(keys)
        delta_colors = min(delta_run/len(cols), delta_max_colors)
        delta_colors = np.linspace(0,delta_run,int((delta_run - 0) / delta_colors + 1))
        delta_run = np.linspace(a,b,int((b-a) / delta_run + 1))
        alpha = util.f_alpha(len(keys))
        linewidth = 0.7
        
        for plot in cols:
            flag = False
            for i, key in enumerate(keys):
                run = self.run_list.get(key)
                alpha_r = alpha
                linewidth_r = linewidth
                if run.id_run=="avg_run":
                    alpha_r = 1  #incremento l'opacità della run media
                    linewidth_r = 1.4
                id = " run "+ str(i+1)
                plot_cols = list(set(plot).intersection(run.indexes()))
                for j, col in enumerate(plot_cols):
                    flag = True
                    if col=="altitude":
                        plt.plot(run.run_data["distance"],run.rescale(col),label=col+id,color=cmap(delta_run[i]+delta_colors[j]),alpha=alpha_r,linewidth=linewidth_r)
                        # prove per il posizionamento delle scritte nel plot ai margini (inizio e fine)
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
                        plt.plot(run.run_data["distance"],run.run_data[col],label=col+id,color=cmap(delta_run[i]+delta_colors[j]),alpha=alpha_r,linewidth=linewidth_r)
                    if col in vis_max:
                        #calculate x and y of max
                        x = run.run_data[col].idxmax()
                        y = run.run_data.at[x, col]
                        x = run.run_data.at[x,"distance"]
                        #TODO studiare il posizionamento adattivo dei testi
                        marginx = 0
                        marginy = 3
                        plt.text(x + marginx, y + marginy, col+"_max : " + str("%.2f" % y))
                        plt.scatter(x, y, s=20)
                # for index in vis_max:
                #     #calculate x and y of max
                #     x = run.run_data[index].idxmax()
                #     y = run.run_data.at[x, index]
                #     x = run.run_data.at[x,"distance"]
                #     marginx = 0
                #     marginy = 3
                #     plt.text(x + marginx, y + marginy, index+"_max : " + str("%.2f" % y))
            if flag==True:
                plt.title("Comparation")
                plt.legend()
                if export_PDF==True:
                    pdf.savefig(bbox_inches='tight', pad_inches=0.5)
                if export_PNG==True:
                    fname = plot[0]
                    for col in plot[1:]:
                        fname = fname + "_" + col
                        fpath = util.joinPath(util.pdfexport_path,fname)
                    plt.savefig(fname=fpath+".png")
                if show==True:
                    plt.show()
                plt.close()
        if export_PDF==True:
            pdf.close()
    
    # def AvgRunRunTime(self): #TODO (maybe)
    #     '''
    #     calculate average run every time a new run is added
    #     '''
            
    def calcAvgRun(self): #Last Version
        '''
        (new version)
        preserve original values of the data in all the races
        '''
        if not self.run_list:
            print("no run in run_list")
            return
        run_list_tmp = copy.deepcopy(self.run_list)
        n_data = float('inf')
        for run in run_list_tmp.values():
            n_data = min(n_data, run.n_data)
        for run in run_list_tmp.values():   #tolgo i dati all'inizio (assumo che i dati alla fine siano sincronizzati)
            run.setBounds(lwbd=0,upbd=run.n_data-n_data) #TODO modificare...
        rlv = list(run_list_tmp.values())
        count_index = {}
        avg_run = Run()
        avg_run.id_run = "avg_run"
        avg_run.run_data = rlv[0].run_data
        avg_run.n_data = n_data
        notnan_disp = 0
        if np.isnan(rlv[0].disp):
            avg_run.disp = 0
        else:
            avg_run.disp = rlv[0].disp
            notnan_disp = notnan_disp + 1
        # print("d:  "+str(avg_run.disp))
        cols = avg_run.indexes()
        cols = np.delete(cols, np.where(cols == "timestamp"))
        cols = np.delete(cols, np.where(cols == "distance"))
        for col in cols:
            count_index[col]=np.zeros(avg_run.n_data)
            for row in range(avg_run.n_data):
                if not np.isnan(avg_run.run_data.iloc[row][col]):
                    count_index[col][row] = 1
        for run in rlv[1:]:
            run_cols = run.indexes()
            run_cols = np.delete(run_cols, np.where(run_cols == "timestamp"))
            run_cols = np.delete(run_cols, np.where(run_cols == "distance"))
            
            if not np.isnan(run.disp):
                avg_run.disp = avg_run.disp + run.disp
                notnan_disp = notnan_disp + 1
            # print(avg_run.disp)

            for col in run_cols:
                if col not in cols:
                    avg_run.run_data[col] = run.run_data[col]
                    count_index[col]=np.zeros(avg_run.n_data)
                    for row in range(avg_run.n_data):
                        if not np.isnan(avg_run.run_data.iloc[row][col]):
                            count_index[col][row] = count_index[col][row] + 1
                else:
                    for row in range(avg_run.n_data):
                        if np.isnan(avg_run.run_data.iloc[row][col]):
                            avg_run.run_data.at[row,col] = run.run_data.at[row,col]
                            if not np.isnan(run.run_data.iloc[row][col]):
                                count_index[col][row] = count_index[col][row] + 1
                        else:
                            if not np.isnan(run.run_data.iloc[row][col]):
                                avg_run.run_data.at[row,col] = avg_run.run_data.at[row,col] + run.run_data.at[row,col]
                                if not np.isnan(run.run_data.iloc[row][col]):
                                    count_index[col][row] = count_index[col][row] + 1
        for col in cols:
            for row in range(avg_run.n_data):
                avg_run.run_data.at[row,col] = avg_run.run_data.at[row,col]/count_index[col][row]
        if notnan_disp==0:
            avg_run.disp = np.nan
        else:
            avg_run.disp = avg_run.disp/notnan_disp
        
        avg_run.calcAvgValues()
        self.addRun(run=avg_run) #,replace=True
    
    def generateCol(self, col="power", avg_value=None, std_value=None, std_perc=None):   #TODO inserire un transitorio (logaritmo,esponenziale,radice)
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

##############################################################################################################
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
        self._model_data = input_values
        self._prediction = output_value
        for run in self.run_list.values():
            for value in all_values:
                if value not in run.indexes():
                    print("no data for \""+value+"\"")
                    return
        
        self._poly = PolynomialFeatures(degree)
        self._model = LinearRegression()

        datasets = self.run_list.values()
        # Addestra e valuta il modello su ciascun dataset
        for i, dataset in enumerate(datasets):
            X = dataset.run_data[input_values].values
            y = dataset.run_data[output_value].values
            
            # per evitare il warning (NON TESTATO)
            # preprocessor = ColumnTransformer(transformers=[('poly', self._poly, dataset.columns)],remainder='passthrough')
            # X_poly = preprocessor.fit_transform(dataset)

            X_poly = self._poly.fit_transform(X)
            self._model.fit(X_poly, y)

            predictions = self._model.predict(X_poly)

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
        simulate self._prediction knowing input_values, using self._model
        '''
        if self._model is None:
            print("No model")
            return
        if input_values is None:
            tmp = {}
            for index in self._model_data:
                tmp[index] = self.generateCol(col=index)
            input_values = pd.DataFrame(tmp)
            
        if not input_values.columns.equals(pd.Index(self._model_data)):
            print("Columns must match")
            return

        simulated_data = input_values
        X_sim_poly = self._poly.transform(simulated_data)
        simulated_data[self._prediction] = self._model.predict(X_sim_poly)
        
        sim_run = Run()
        sim_run.id_run = "sim_run"
        sim_run.n_data = self.run_list["avg_run"].n_data
        sim_run.run_data = simulated_data
        sim_run.run_data["distance"] = self.run_list["avg_run"].run_data["distance"]
        sim_run.calcAvgValues()
        self.addRun(sim_run)
        
        if plot==True:
            tmp = np.arange(0,len(simulated_data[self._prediction]))
            plt.scatter(tmp, simulated_data[self._prediction], color='red', label='Simulated '+self._prediction)
            for index in self._model_data:
                plt.scatter(tmp, simulated_data[index], color='orange', label=index)
            plt.xlabel('distance')
            plt.ylabel(self._prediction+' Profile')
            plt.title('Simulation Results')
            plt.legend()
            plt.show()
            
            # plt.scatter(simulated_data['distance'], simulated_data[self._prediction], color='red', label='Simulated '+self._prediction)
            # for index in self._model_data:
            #     plt.scatter(simulated_data['distance'], simulated_data[index], color='orange', label=index)
            # plt.xlabel('distance')
            # plt.ylabel(self._prediction+' Profile')
            # plt.title('Simulation Results')
            # plt.legend()
            # plt.show()
            
        if export==True:
            pass


    # def export(self): #TO DELETE
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


