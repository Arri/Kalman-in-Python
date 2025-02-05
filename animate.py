##########################################
# Animation of kalman filter outputs

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.animate as animation
import matplotlib
import csv

DATAPATH = '/home/pi/Projects/kalman/resultFile.csv'


def run():
	with open(DATAPATH, 'r') as fileC:
		readdat = csv.reader(fileC, delimiter= ',')
		for [xin1, xin2, xin3, xout1, xout2, xout3] in readdat:
			print(xout1, xout2, xout3) 

if __name__=="__main__":
	run()
