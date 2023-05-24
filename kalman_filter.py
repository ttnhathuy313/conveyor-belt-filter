import cv2 as cv 
import numpy as np

class KalmanFilter:
    
    def __init__(self):
        self.variance_measurement = 0.0005300125234210817
        self.predict_variance = 0.0005300125234210817
        self.kalman_gain = 0.0
        self.predict_velocity = 0.20
        self.current_velocity = 0.20
        self.cnt = 0
        
    def state_update(self, measurement):
        self.kalman_gain = self.predict_variance / (self.predict_variance + self.variance_measurement)
        self.current_velocity = self.predict_velocity + self.kalman_gain * \
        (measurement - self.predict_velocity)
        self.predict_variance = (1 - self.kalman_gain) * self.predict_variance
        self.predict()
        return self.current_velocity
    
    def predict(self):
        self.predict_velocity = self.current_velocity
        self.predict_variance = self.predict_variance
        
        
    