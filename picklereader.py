__author__ = 'wilmer'
try:
    import mkl
    have_mkl = True
    print("Running with MKL Acceleration")
except ImportError:
    have_mkl = False
    print("Running with normal backends")

import glob, os
import numpy as np
from scipy.optimize import minimize, rosen, rosen_der
import time
import math
import pylab
import matplotlib.pyplot as plt
import pickle
import sys

WholeCircle = False
kappasize = 16

def load(filename):
    with open(filename, "rb") as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                print('error')
                break

class something(object):
    def __init__(self, name, value):
        self.name = name
        self.value = value
l = [1,2,3,4]
b = something('obama', 42)
r = "hi"
x0 = [1.3, 0.7, 0.8, 1.9, 1.2]
res = minimize(rosen, x0, method='Nelder-Mead')

with open('somefile.pkl', 'wb') as output:
    pickle.dump([l, b, r, x0, res], output, pickle.HIGHEST_PROTOCOL)
output.close()
with open('somefile.pkl', 'rb') as input:
    items = pickle.load(input)
input.close()
print(items[0], items[2], items[1].name, items[4])

print('todo bien')

for iter in np.arange(0, 5, 0.5):
    PIK = "/home/wilmer/Dropbox/Research/VMAT/VMATwPenCode/outputGraphics/pickle-C-" + str(iter) + "-WholeCirCle-" + str(WholeCircle) + "-Kappa-" + str(kappasize) + "-save.dat"
    #PIK = "/home/wilmer/Dropbox/Research/VMAT/VMATwPenCode/outputGraphics/pickle-C-0-WholeCirCle-False-Kappa-16-save.dat"
    print(PIK)
    try:
        with open(PIK, "rb") as f:
            items = pickle.load(f)
            print('items:', items[0], items[1])
            #for item in items:
        f.close()
    except IOError:
        sys.exit('Error:no file found')
