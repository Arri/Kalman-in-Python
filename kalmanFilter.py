##!/usr/bin/env python
# ################################################################################################################
# Author: Arasch U. Lagies 
# Last Update: 12/19/2019
#
# Based on: https://github.com/zziz/kalman-filter and wickipedia (https://en.wikipedia.org/wiki/Kalman_filter)
# ################################################################################################################
# Fk, the state-transition model;
# Hk, the observation model;
# Qk, the covariance of the process noise;
# Rk, the covariance of the observation noise;
# and sometimes Bk, the control-input model, for each time-step, k, as described below.
# ################################################################################################################
# packages to install:
# conda install -c conda-forge ffmpeg
#
# Call with > python kalmanFilter6.py --dataPath Path-to-the-Coordinates

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import matplotlib.animation as animation
import matplotlib
import argparse

measurements = []
predict_x = []
predict_y = []

path = "./random.txt"

class KalmanFilter(object):
    def __init__(self, F = None, B = None, H = None, Q = None, R = None, P = None, x0 = None, y0 = None):
        if(F is None or H is None):
            raise ValueError("Set proper system dynamics.")

        self.n = F.shape[1]
        self.m = H.shape[1]

        self.F = F                      # The state-transition model
        self.H = H                      # The observation model (maps the true state space into the observed space )
        self.B = 0 if B is None else B  # The control-input model (=0 here ==> the applied model is velocity=const.)
        self.Q = np.eye(self.n) if Q is None else Q     # The covariance of the process noise
        self.R = np.eye(self.n) if R is None else R     # The covariance of the observation noise
        self.P = np.eye(self.n) if P is None else P     # a posteriori error covariance matrix (a measure of the estimated accuracy of the state estimate)
        self.x = np.zeros((self.n, 1)) if x0 is None else x0    # x-coordinate
        self.y = np.zeros((self.n, 1)) if y0 is None else y0    # y-coordinate

    def predict(self, u = 0, v = 0):
        self.x = np.dot(self.F, self.x) + np.dot(self.B, u)
        self.y = np.dot(self.F, self.y) + np.dot(self.B, v)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        # print(f"-- u.shape = {u} --- v.shape = {v} --- x.shape = {self.x.shape} --- y.shape = {self.y.shape} --- P.shape = {self.P.shape} --- np.dot(self.F, self.x) shape = {np.dot(self.F, self.x).shape} --- np.dot(self.B, u) shape = { np.dot(self.B, u).shape}.")
        return self.x, self.y

    def update(self, zx, zy):
        yx = zx - np.dot(self.H, self.x)
        yy = zy - np.dot(self.H, self.y)
        #print(f" shape of yx = {yx.shape}, shape of zx = {zx.shape}")
        S = self.R + np.dot(self.H, np.dot(self.P, self.H.T))
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        #print(f"the value of S is {S} --- applying linalg.inv on it is {np.linalg.inv(S)} and the shape of K is {K.shape}")
        self.x = self.x + np.dot(K, yx)
        self.y = self.y + np.dot(K, yy)
        I = np.eye(self.n)
        #print(f" I is {I}, and the shape of KxH is {np.dot(K, self.H).shape}...")
        self.P = np.dot(np.dot(I - np.dot(K, self.H), self.P), 
            (I - np.dot(K, self.H)).T) + np.dot(np.dot(K, self.R), K.T)

class tracking(KalmanFilter, object):
    def __init__(self, path, startCol, endCol):
        self.measurements = []
        self.predict_x = []
        self.predict_y = []
        self.File = Path(path).resolve()
        self.startCol = startCol
        self.endCol   = endCol

    def animate(self, i, plt):
        self.xp = self.predict_x[i]
        self.yp = self.predict_y[i]
        self.xm = self.measurements.values[i,0]
        self.ym = self.measurements.values[i,1]   
        #print("self.predict_x = " + str(self.predict_x))
        self.h2.set_data(np.append(self.h2.get_xdata(), self.xp), np.append(self.h2.get_ydata(), self.yp))
        self.h1.set_data(np.append(self.h1.get_xdata(), self.xm), np.append(self.h1.get_ydata(), self.ym))
        self.h1.set_label('Measurements')
        self.h2.set_label('Kalman Filter Prediction')
    
        legend = plt.legend()
        #plt.draw()

        legend.remove()
        legend = plt.legend()
        return self.h1, self.h2

    def track(self):
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=20, metadata=dict(artist='Me'), bitrate=1800)

        # Read the measurement file...:
        try:        # Try reading in Excel format...
            df = pd.read_excel(self.File, sheet_name=0)
        except:     # Try reading if text format...
            df = pd.read_csv(self.File, sep="\t", header=None)

        dt = 1.0/60
        F = np.array([[1, dt, 0], [0, 1, dt], [0, 0, 1]])                       # the state-transition model
        H = np.array([1, 0, 0]).reshape(1, 3)                                   # the observation model
        Q = np.array([[0.05, 0.05, 0.0], [0.05, 0.05, 0.0], [0.0, 0.0, 0.0]])   # the covariance of the process noise
        R = np.array([0.5]).reshape(1, 1)                                       # the covariance of the observation noise
        kf = KalmanFilter(F = F, H = H, Q = Q, R = R)

        try:
            self.measurements = df.loc[:,self.startCol:self.endCol]             # Read in the x- and y-Koordinates
        except:
            self.measurements = df[df.columns[self.startCol:self.endCol+1]]     # Read in the x- and y-Koordinates

        for x,y in self.measurements.values:
            #print("x,y = " + str(x) + " " + str(y))
            #kf.update(x,y)
            pr_x, pr_y = kf.predict()
            #print(f"pr_x size is {pr_x.shape} --- pr_y size is {pr_y.shape}")
            print(f" Result shapes: x = {np.dot(H,  pr_x).shape}  and for y = {np.dot(H,  pr_y).shape}")
            self.predict_x.append(np.dot(H,  pr_x)[0][0])
            self.predict_y.append(np.dot(H,  pr_y)[0][0])
            kf.update(x,y)

        length = self.measurements.shape
        length = length[0]

        #fig = plt.figure(figsize=(10,6))
        fig, _ = plt.subplots()

        #plt.ion()
        plt.grid()
        plt.axis([200,400,200,400])
        self.h1, = plt.plot([], [], label='Measurements')
        self.h2, = plt.plot([], [], label='Kalman Filter Prediction')

        ani = matplotlib.animation.FuncAnimation(fig, self.animate, frames=length, \
                                        fargs=(plt, ), \
                                        interval=100, blit=True, repeat=True)
        #ani.save('kalman.mp4', writer=writer)
        plt.show()
        
        input("Press [enter] to continue.")

def run(dataPath):
    startCol = 2        #" x"
    endCol   = 3        #" y"
    track = tracking(dataPath, startCol, endCol)
    track.track()

if __name__ == '__main__':
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataPath", type=str, required=False,
    default=path, help="path to data file with tracking x/y coordinates...")
    args = vars(ap.parse_args())
    dataPath = args["dataPath"]

    # Check if the file exists...
    File = Path(dataPath)
    if not File.is_file():
        print("[ERROR] Could not find the file with the x/y coordinates...")
        print("[NOTE] Use the input argument --dataPath to provide the correct path to the coordinates...")
        exit(0)

    run(dataPath)
    print("Done")