
class Vehicle:
    '''Contains information of a vehicle (bike)'''
    def __init__(self,name):
        self.name = name
        self.chassis_weight = None   #peso telaio
        self.hull_weight = None   #peso carena
        self.frontal_area = None
        self.inertia = None
        self.leg_traction = None
        self.crank = None   #pedivella

    def getInfoFromExcel(self):
        pass

    def setInfo(self,chassis_weight,hull_weight,frontal_area,inertia,leg_traction,crank):
        self.chassis_weight = chassis_weight   #peso telaio
        self.hull_weight = hull_weight   #peso carena
        self.frontal_area = frontal_area
        self.inertia = inertia
        self.leg_traction = leg_traction
        self.crank = crank

    def setName(self,name):
        self.name = name

class GearBox:
    def __init__(self, gear_box):
        '''
        gear_box: List of gear
        chainring: Integer
        sec_ratio: pair of Integers
        '''
        self.gear_box = gear_box
        self.chainring = None   #moltiplica
        self.sec_ratio = None   #secondary ratio / rimando finale
    
    def setInfo(self, chainring, sec_ratio):
        self.chainring = chainring
        self.sec_ratio = sec_ratio
        
    def getInfoFromExcel(self):
        pass

class Wheel:
    '''Contains information of a wheel (of a bike)'''
    def __init__(self,tyre):
        self.tyre = tyre
        self.pressure = None
        self.radius = None
        self.rolling_circum = None   #circonferenza di rotolamento
        self.inertia = None

    def getInfoFromExcel(self):
        pass

    def setInfo(self,pressure,radius,rolling_circum,inertia):
        self.pressure = pressure
        self.radius = radius
        self.rolling_circum = rolling_circum
        self.inertia = inertia

    def setTyre(self,name):
        self.name = name

class Driver:
    '''Contains information of a driver (of a bike)'''
    def __init__(self,name):
        self.name = name
        self.weight = None

    def getInfoFromExcel(self):
        pass

    def setInfo(self,weight):
        self.weight = weight

    def setName(self,name):
        self.name = name

class AtmConditions:
    '''Contains information of atmospheric conditions'''
    def __init__(self):
        self.temperature = None
        self.pressure = None
        self.humidity = None
        self.wind = None
        self.angle = None   #clockwise?

    def setInfo(self,temperature,pressure,humidity,wind,angle):
        self.temperature = temperature
        self.pressure = pressure
        self.humidity = humidity
        self.wind = wind
        self.angle = angle

    def getInfoFromCsv(self):
        pass

class BikeInfo:
    '''Contains all information of Bike and Driver'''
    def __init__(self, vehicle_obj, driver_obj, wheels_obj, gear_box):
        self.bike = vehicle_obj
        self.driver = driver_obj
        self.wheels = wheels_obj
        self.gear_box = gear_box   #sprocket pack / pacco pignoni

    def getInfoFromExcel(self):
        self.bike.getInfoFromExcel()
        self.driver.getInfoFromExcel()
        self.wheels.getInfoFromExcel()
        self.gear_box.getInfoFromExcel()
        

