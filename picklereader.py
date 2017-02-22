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
from scipy.optimize import minimize
import time
import math
import pylab
import matplotlib.pyplot as plt
import pickle
import sys
import itertools

WholeCircle = False
kappasize = 16
myfolder = '/home/wilmer/Dropbox/Research/VMAT/VMATwPenCode/outputGraphics/'

## The next function plots aperture shapes
def plotApertures(C, numbeams, M, N, YU, llist, rlist, x, myfolder):
    magnifier = 100
    ## Plotting apertures
    xcoor = math.ceil(math.sqrt(numbeams))
    ycoor = math.ceil(math.sqrt(numbeams))
    nrows, ncols = M,N
    print('numbeams', numbeams)
    for mynumbeam in range(0, numbeams):
        l = llist[mynumbeam]
        r = rlist[mynumbeam]
        ## Convert the limits to hundreds.
        #for posn in range(0, len(l)):
            #l[posn] = int(magnifier * l[posn])
            #r[posn] = int(magnifier * r[posn])
        image = -1 * np.ones(magnifier * nrows * ncols)
        image = image.reshape((nrows, magnifier * ncols))
        for i in range(0, M):
            image[i, l[i]:(r[i]-1)] = x[mynumbeam]
        image = np.repeat(image, magnifier, axis = 0) # Repeat. Otherwise the figure will look flat like a pancake
        image[0,0] = YU # In order to get the right list of colors
        # Set up a location where to save the figure
        fig = plt.figure(1)
        plt.subplot(ycoor,xcoor, mynumbeam + 1)
        cmapper = plt.get_cmap("autumn_r")
        cmapper.set_under('black', 1.0)
        plt.imshow(image, cmap = cmapper, vmin = 0.0, vmax = YU)
        plt.axis('off')
    fig.savefig(myfolder + 'plotofaperturesFromPickle'+ str(C) + '.png')

# The next function prints DVH values
def printresults(fullMaskValue, currentDose, numstructs, allNames, myfolder, C):
    numzvectors = 1
    maskValueFull = np.array([int(i) for i in fullMaskValue])
    print('Starting to Print Results')
    for i in range(0, numzvectors):
        zvalues = currentDose
        maxDose = max([float(i) for i in zvalues])
        dose_resln = 0.1
        dose_ub = maxDose + 10
        bin_center = np.arange(0, dose_ub, dose_resln)
        # Generate holder matrix
        dvh_matrix = np.zeros((numstructs, len(bin_center)))
        # iterate through each structure
        for s in range(0, numstructs):
            allNames[s] = allNames[s].replace("_VOILIST.mat", "")
            doseHolder = sorted(zvalues[[i for i, v in enumerate(maskValueFull & 2 ** s) if v > 0]])
            if 0 == len(doseHolder):
                continue
            histHolder = []
            carryinfo = 0
            histHolder, garbage = np.histogram(doseHolder, bin_center)
            histHolder = np.append(histHolder, 0)
            histHolder = np.cumsum(histHolder)
            dvhHolder = 1-(np.matrix(histHolder)/max(histHolder))
            dvh_matrix[s,] = dvhHolder
    myfig = pylab.plot(bin_center, dvh_matrix.T, linewidth = 2.0)
    plt.grid(True)
    plt.xlabel('Dose Gray')
    plt.ylabel('Fractional Volume')
    plt.title('Number of beams: ' + str(numbeams))
    plt.legend(allNames, fontsize=8)
    plt.savefig(myfolder + 'full-DVH-VMAT-C-' + str(C) + '.png')
    plt.close()
    voitoplot = [0, 18, 23, 17, 2, 8]
    dvhsub2 = dvh_matrix[voitoplot,]
    myfig2 = pylab.plot(bin_center, dvhsub2.T, linewidth = 1.0, linestyle = '--')
    plt.grid(True)
    plt.xlabel('Dose Gray')
    plt.ylabel('Fractional Volume')
    plt.title('Number of beams: ' + str(numbeams))
    plt.legend([allNames[i] for i in voitoplot], fontsize=10)
    plt.savefig(myfolder + 'DVH-Subplot-VMAT-C-' + str(C) + '.png')
    plt.close()

qualities = []
Cvalues = []
a = np.arange(0.5, 3.5, 1.0)
b = np.arange(15.55, 55.55, 10)
for iter in itertools.chain(a,b):
    PIK = myfolder + "pickle-C-" + str(iter) + "-WholeCirCle-" + str(WholeCircle) + "-Kappa-" + str(kappasize) + "-save.dat"
    print(PIK)
    try:
        with open(PIK, "rb") as f:
            items = pickle.load(f)
            numbeams = items[0]
            x = items[1]
            C = items[2]
            C2 = items[3]
            C3 = items[4]
            vmax = items[5]
            speedlim = items[6]
            RU = items[7]
            YU = items[8]
            M = items[9]
            N = items[10]
            llist = items[11]
            rlist = items[12]
            fullMaskValue = items[13]
            currentDose = items[14]
            currentIntensities = items[15]
            numstructs = items[16]
            allNames = items[17]
            objectiveValue = items[18]
            quadHelperThresh = items[19]
            quadHelperOver = items[20]
            quadHelperUnder = items[21]
            regionIndices = items[22]
            targets = items[23]
            oars = items[24]
            items = None
        f.close()
    except IOError:
        sys.exit('Error:no file found')
    #printresults(fullMaskValue, currentDose, numstructs, allNames, myfolder, C)
    #plotApertures(C, numbeams, M, N, YU, llist, rlist, x, myfolder)
    qualities.append(objectiveValue)
    Cvalues.append(C)
    print('objective Value:', objectiveValue)
plt.plot(Cvalues, qualities, 'ro')
plt.suptitle('Objective Value and C', fontsize=20)
plt.xlabel('C = Shape penalization coefficient', fontsize=18)
plt.ylabel('Optimal Objective Value', fontsize=16)
plt.show()