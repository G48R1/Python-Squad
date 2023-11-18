import conditions as cond

class Run:
    '''
    '''
    def __init__(self):
        self.atm_cond = None
        self.bike_info = None
        self.run_data = None

    def setAtmCond(self, atm_cond):
        self.stm_cond = atm_cond
    
    def setBikeInfo(self, bike_info):
        self.bike_info = bike_info

    def readRun(self, file_name):
        with open(file_name,'r') as file:
            
