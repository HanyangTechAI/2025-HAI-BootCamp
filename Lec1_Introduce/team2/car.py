class Car:
    def __init__(self, registrationNum, year: int, licenseNumber: str):
        self.registrationNum = registrationNum
        self.year = year
        self.licenseNumber = licenseNumber
    
    def moveForward():
        print("The car is moving forward now.")

    def moveBackward():
        print("The car is moving backward now.")
    
    def stop():
        print("The car is stopped")