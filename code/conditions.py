import util
import numpy as np

list_of_object = {"Vehicle": "bike", "Driver": "driver", "Wheels": "wheels", "GearBox": "gear_box"}

class Vehicle:
    '''Contains information of a vehicle (bike)'''
    def __init__(self,name=None,chassis_weight=None,hull_weight=None,frontal_area=None,inertia=None,leg_traction=None,crank=None):
        '''name: String'''
        self.name = name
        self.chassis_weight = chassis_weight   #peso telaio
        self.hull_weight = hull_weight   #peso carena
        self.frontal_area = frontal_area
        self.inertia = inertia
        self.leg_traction = leg_traction
        self.crank = crank   #pedivella
        self.string_attribute = ["name"]

    def getInfoFromMatrix(self, matrix):
        '''
        matrix: 2D Array-like (attribute, value)
        '''
        for attribute, value in matrix:
            if value != '':
                if attribute not in self.string_attribute:
                    value = float(value)
            try:
                setattr(self, attribute, value)
            except AttributeError:
                print(f"Object has no attribute {attribute}")
                
    def getInfoFromDict(self, dictionary={}):
        '''
        dict: Dict Object
        '''
        if not dictionary=={}:
            for key in dictionary.keys():
                try:
                    setattr(self, key, dictionary.get(key))
                except AttributeError:
                    print(f"Object has no attribute {key}")

    
    def setInfo(self,name,chassis_weight,hull_weight,frontal_area,inertia,leg_traction,crank):
        '''
        chassis_weight: Float
        hull_weight: Float
        frontal_area: Float
        inertia: Float
        leg_traction: Bool
        crank: Float (length)'''
        self.name = name
        self.chassis_weight = chassis_weight   #peso telaio
        self.hull_weight = hull_weight   #peso carena
        self.frontal_area = frontal_area
        self.inertia = inertia
        self.leg_traction = leg_traction
        self.crank = crank

    def setName(self,name):
        '''name: String'''
        self.name = name

class GearBox:
    def __init__(self, gear_box=None, chainring=None, sec_ratio=None):
        '''
        gear_box: List of gear
        chainring: Integer
        sec_ratio: pair of Integers
        '''
        self.gear_box = gear_box
        self.chainring = chainring   #moltiplica
        self.sec_ratio = sec_ratio   #secondary ratio / rimando finale
        self.string_attribute = []
        self.list_attribute = ["gear_box", "sec_ratio"]

    def setInfo(self, gear_box, chainring, sec_ratio):
        '''
        chainring: Int (number of theet)
        sec_ratio: 2D Iterable (number of theet, es. [greatest, lowest])
        '''
        self.gear_box = gear_box
        self.chainring = chainring
        self.sec_ratio = sec_ratio
        
    def getInfoFromMatrix(self, matrix):
        '''
        matrix: 2D Array-like (attribute, value)
        '''
        for attribute, value in matrix:
            if value != '':
                if attribute not in self.string_attribute:
                    if attribute in self.list_attribute:
                        value = [int(num) for num in value.split(',')]
                    else:
                        value = int(value)
            try:
                setattr(self, attribute, value)
            except AttributeError:
                print(f"Object has no attribute {attribute}")
                
    def getInfoFromDict(self, dictionary={}):
        '''
        dict: Dict Object
        '''
        if not dictionary=={}:
            for key in dictionary.keys():
                try:
                    setattr(self, key, dictionary.get(key))
                except AttributeError:
                    print(f"Object has no attribute {key}")

class Wheels:
    '''Contains information of a wheel (of a bike)'''
    def __init__(self,tyre=None,pressure=None,radius=None,rolling_circum=None,inertia=None):
        self.tyre = tyre
        self.pressure = pressure
        self.radius = radius
        self.rolling_circum = rolling_circum   #circonferenza di rotolamento
        self.inertia = inertia
        self.string_attribute = ["tyre"]


    def setInfo(self,tyre,pressure,radius,rolling_circum,inertia):
        '''
        pressure: Float
        radius: Float
        rolling_circum: Float
        inertia: Float
        '''
        self.tyre = tyre
        self.pressure = pressure
        self.radius = radius
        self.rolling_circum = rolling_circum
        self.inertia = inertia

    def setTyre(self,name):
        self.name = name

    def getInfoFromMatrix(self, matrix):
        '''
        matrix: 2D Array-like (attribute, value)
        '''
        for attribute, value in matrix:
            if value != '':
                if attribute not in self.string_attribute:
                    value = float(value)
            try:
                setattr(self, attribute, value)
            except AttributeError:
                print(f"Object has no attribute {attribute}")
                
    def getInfoFromDict(self, dictionary={}):
        '''
        dict: Dict Object
        '''
        if not dictionary=={}:
            for key in dictionary.keys():
                try:
                    setattr(self, key, dictionary.get(key))
                except AttributeError:
                    print(f"Object has no attribute {key}")

class Driver:
    '''Contains information of a driver (of a bike)'''
    def __init__(self,name=None,weight=None):
        self.name = name
        self.weight = weight
        self.string_attribute = ["name"]

    def setInfo(self,name,weight):
        '''weight: Float'''
        self.name = name
        self.weight = weight

    def setName(self,name):
        '''name: String'''
        self.name = name

    def getInfoFromMatrix(self, matrix):
        '''
        matrix: 2D Array-like (attribute, value)
        '''
        for attribute, value in matrix:
            if value != '':
                if attribute not in self.string_attribute:
                    value = float(value)
            try:
                setattr(self, attribute, value)
            except AttributeError:
                print(f"Object has no attribute {attribute}")
                
    def getInfoFromDict(self, dictionary={}):
        '''
        dict: Dict Object
        '''
        if not dictionary=={}:
            for key in dictionary.keys():
                try:
                    setattr(self, key, dictionary.get(key))
                except AttributeError:
                    print(f"Object has no attribute {key}")

class AtmConditions:
    '''Contains information of atmospheric conditions'''
    def __init__(self,temperature=None,pressure=None,humidity=None,wind=None,angle=None):
        self.temperature = temperature
        self.pressure = pressure
        self.humidity = humidity
        self.wind = wind
        self.angle = angle   #clockwise?

    def setInfo(self,temperature,pressure,humidity,wind,angle):
        '''
        temperature: Iterable of Float
        pressure: Iterable of Float
        humidity,: Iterable of Float
        wind: Iterable of Float
        angle: Iterable of Float
        '''
        self.temperature = temperature
        self.pressure = pressure
        self.humidity = humidity
        self.wind = wind
        self.angle = angle

    def getInfoFromCsv(self):
        pass

    def getInfoFromDict(self, dictionary={}):
        '''
        dict: Dict Object
        '''
        if not dictionary=={}:
            for key in dictionary.keys():
                try:
                    setattr(self, key, dictionary.get(key))
                except AttributeError:
                    print(f"Object has no attribute {key}")

class BikeInfo:
    '''Contains all information of Bike and Driver'''
    def __init__(self, vehicle_obj=Vehicle(), driver_obj=Driver(), wheels_obj=Wheels(), gear_box_obj=GearBox()):
        self.bike = vehicle_obj
        self.driver = driver_obj
        self.wheels = wheels_obj
        self.gear_box = gear_box_obj   #sprocket pack / pacco pignoni

    def setInfo(self, vehicle_obj, driver_obj, wheels_obj, gear_box_obj):
        self.bike = vehicle_obj
        self.driver = driver_obj
        self.wheels = wheels_obj
        self.gear_box = gear_box_obj

    def getInfoFromExcel(self, csv_file):
        '''
        csv_file: String (path)
        '''
        data, dummy = util.readCsvFile(csv_file, delimiter=';')
        
        for row in data:
            attribute = row[0]
            value = row[1]
            if str(attribute).strip()=="":
                pass
            elif attribute in list_of_object.keys():
                obj_name = list_of_object[attribute]
                obj = getattr(self,obj_name)
            else:
                obj.getInfoFromMatrix([[attribute,value]])
        

