
class Vehicle:
'''
Contains information of a vehicle (bike)
'''
    def __init__(self,name):
        self.name = name
        self.chassis_weight = None   #peso telaio
        self.hull_weight = None   #peso carena
        self.frontal_area = None
        self.inertia = None
        self.leg_traction = None

    def getInfoFromExcel(self):
        pass

    def setInfo(self,chassis_weight,hull_weight,frontal_area,inertia,leg_traction):
        self.chassis_weight = chassis_weight   #peso telaio
        self.hull_weight = hull_weight   #peso carena
        self.frontal_area = frontal_area
        self.inertia = inertia
        self.leg_traction = leg_traction

    def setName(self,name):
        self.name = name


class Wheel:
'''
Contains information of a wheel (of a bike)
'''
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
'''
Contains information of a driver (of a bike)
'''
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
'''
Contains information of atmospheric conditions
'''
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
        self.angle = angle   #clockwise?

    def getInfoFromCsv(self):
        pass

class BikeInfo:
'''
Contains all information of Bike and Driver
'''
    def __init__(self,vehicle_name,driver_name,wheels_tyre):
        self.bike = Vehicle(vehicle_name)
        self.driver = Driver(driver_name)
        self.wheels = Wheel(wheels_tyre)

    def getInfoFromExcel(self):
        self.bike.getInfoFromExcel()
        self.driver.getInfoFromExcel()
        self.wheels.getInfoFromExcel()


