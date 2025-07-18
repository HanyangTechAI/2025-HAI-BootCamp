class Car:
    def __init__(self,registrationNum, year, licenseNumber):
        self.registrationNum=registrationNum
        self.year=year
        self.licenseNumber=licenseNumber
    
    def moveForward(self) :
        print("move forward")

    def moveBackward(self):
        print("move backward")

    def stop(self):
        print("stop")

    def turnRight(self):
        print("turn right")

    def turnLeft(self):
        print("turn left")