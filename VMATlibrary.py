

import glob, os
#import pyipopt
import numpy as np
import scipy.io as sio
from scipy import sparse
from scipy.optimize import minimize
import time
import math
print('read here')

class region:
    """ Contains all information relevant to a particular region"""
    index = int()
    sizeInVoxels = int()
    indices = np.empty(1, dtype=int)
    fullIndices = np.empty(1,dtype=int)
    target = False
    # Class constructor
    def __init__(self, iind, iindi, ifullindi, itarget):
        self.index = iind
        self.sizeInVoxels = len(iindi)
        self.indices = iindi
        self.fullIndices = ifullindi
        self.target = itarget

class vmat_class:
    # constants particular to the problem
    numX = 0 # num beamlets
    numvoxels = int() #num voxels (small voxel space)
    numstructs = 0 # num of structures/regions
    numoars = 0 # num of organs at risk
    numtargets = 0 # num of targets
    numbeams = 0 # num of beams
    totaldijs = 0 # num of nonzeros in Dij matrix
    nnz_jac_g = 0
    
    # vectors
    beamletsPerBeam = [] # number of beamlets per beam 
    dijsPerBeam = [] # number of nonzeroes in Dij per beam
    maskValue = [] #non-overlapping mask value per voxel
    fullMaskValue = [] # complete mask value per voxel
    regionIndices = [] # index values of structures in region list (should be 0,1,etc)
    targets = [] # region indices of structures (from region vector)
    oars = [] # region indices of oars
    regions = [] # vector of regions (holds structure information)
    objectiveInputFiles = [] # vector of data input files for objectives
    constraintInputFiles = [] # vector of data input files for constraints
    algOptions = [] # vector of data input for algorithm options
    functionData = []
    voxelAssignment = []
    
    # varios folders
    outputDirectory = ""# given by the user in the first lines of *.py
    dataDirectory = ""

    # dose variables
    currentDose = [] # dose variable
    currentIntensities = []
    
    caligraphicC = []
    # this is the intersection of all beamlets that I am dealing with
    xinter = []
    yinter = []
    
    xdirection = []
    ydirection = []
    
    llist = []
    rlist = []

    mygradient = []
    # data class function
    def calcDose(self):
        print('entre')
        # self.currentDose = self.Dmat.transpose() * newIntensities
        self.currentDose = np.zeros(251897, dtype = float)
        for i in self.caligraphicC:
            ThisDlist = Dlist[i]
            for m in range(0, llist[i]):
                ThisDlist[:, range((m * len(data.yinter)),(m * len(data.yinter)) + data.llist[i][m])] = 0.0
                #ThisDlist[((m * numvoxels):data.llist[i][m])] 
            self.currentDose += ThisDlist * np.repeat(self.currentIntensities[i], Dlist[i].shape[1], axis = 0)

        oDoseObj = self.currentDose - quadHelperThresh
        oDoseObjCl = (oDoseObj > 0) * oDoseObj
        oDoseObj = (oDoseObj > 0) * oDoseObj
        oDoseObj = oDoseObj * oDoseObj * quadHelperOver
    
        uDoseObj = quadHelperThresh - self.currentDose
        uDoseObjCl = (uDoseObj > 0) * uDoseObj
        uDoseObj = (uDoseObj > 0) * uDoseObj
        uDoseObj = uDoseObj * uDoseObj * quadHelperUnder
        objectiveValue = sum(oDoseObj + uDoseObj)
    
        oDoseObjGl = 2 * oDoseObjCl * quadHelperOver
        uDoseObjGl = 2 * uDoseObjCl * quadHelperUnder
            # Wilmer. Is this right?
        self.mygradient = 2 * (oDoseObjGl - uDoseObjGl)
   
    # default constructor
    def __init__(self):
        self.numX = 0

########## END OF CLASS DECLARATION ###########################################
