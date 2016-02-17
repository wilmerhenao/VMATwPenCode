#!/usr/bin/env python

__author__ = 'wilmer'
try:
    import mkl
    have_mkl = True
    print("Running with MKL Acceleration")
except ImportError:
    have_mkl = False
    print("Running with normal backends")

import glob, os
#import pyipopt
import numpy as np
import scipy.io as sio
from scipy import sparse
from scipy.optimize import minimize
import time
import math

rootFolder = '/media/wilmer/datadrive'
# rootFolder = '/home/wilmer/Documents/Troy_BU'
readfolder = rootFolder + '/Data/DataProject/HN/'
readfolderD = readfolder + 'Dij/'
outputfolder = '/home/wilmer/Dropbox/Research/VMAT/output/'
degreesep = 60 # How many degrees in between separating neighbor beams.
objfile = '/home/wilmer/Dropbox/IpOptSolver/TestData/HNdata/objectives/obj1.txt'
structurefile = '/home/wilmer/Dropbox/IpOptSolver/TestData/HNdata/structureInputs.txt'
algfile = '/home/wilmer/Dropbox/IpOptSolver/TestData/HNdata/algInputsWilmer.txt'
mm3voxels = rootFolder + '/Data/DataProject/HN/hn3mmvoxels.mat'
# The 1 is subtracted at read time so the user doesn't have to do it everytime
priority = [7, 24, 25, 23, 22, 21, 20, 16, 15, 14, 13, 12, 10, 11, 9, 4, 3, 1, 2, 17, 18, 19, 5, 6, 8]
priority = (np.array(priority)-1).tolist()
mylines = [line.rstrip('\n') for line in open('/home/wilmer/Dropbox/Research/VMAT/VMATwPenCode/beamAngles.txt')]

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
    objectiveValue = float("inf")
    
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
    notinC = [] # List of apertures not yet selected
    caligraphicC = [] # List of apertures already selected
    
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
    scipygradient = []
    GradientIntermediate = 0.0

    # data class function
    def calcDose(self):
        # self.currentDose = self.Dmat.transpose() * newIntensities
        self.currentDose = np.zeros(self.numvoxels, dtype = float)
        # gradhelper will have a dimension that is numvoxels x numbeams
        # gradhelper = Dlist[0].sum(axis=0).transpose()
        # for i in range(1, self.numbeams):
            # Sum all the rows.
        #    gradhelper = np.hstack((gradhelper, Dlist[i].sum(axis=0).transpose()))
        # gradhelpershade = gradhelper * 0.0
        gradhelper = np.matrix(np.zeros((self.numvoxels, self.numbeams)))
        if len(self.caligraphicC) != 0:
            for i in self.caligraphicC:
                ThisDlist = Dlist[i]
                #WILMER Change to:
                ThisDlistshader = ThisDlist * 0.0
                #ThisDlistshader = ThisDlist * 1.0
                leftlimits = 0
                for m in range(0, len(data.llist[0])):
                    # Find geographical values of llist and rlist.
                    # Find geographical location of the first row.
                    geolocX = data.xinter[m]
                    # Find all possible locations of beamlets in this row according to geographical location
                    indys = np.where(geolocX == data.xdirection[i])
                    ys = data.ydirection[i][indys]
                    validbeamlets = np.in1d(data.yinter, ys)
                    # After next line, validbeamlets contains all beamlets that are "fair game" in this row.
                    validbeamlets = np.array(range(0, len(data.yinter)))[validbeamlets]
                    # First index in this row
                    indleft = data.llist[i][m] - min(validbeamlets) + 1 + leftlimits
                    indright = data.rlist[i][m] - min(validbeamlets) - 1 + leftlimits
                    # Keep the location of the le)ftmost leaf
                    leftlimits = leftlimits + len(validbeamlets)
                    if (indleft < indright + 1):
                        self.currentDose += ThisDlist[[ij for ij in range(indleft, indright + 1)],:].transpose()  * np.repeat(self.currentIntensities[i], (indright - indleft + 1), axis = 0)
                        # WILMER Change to:
                        diagmaker = np.zeros(ThisDlistshader.shape[0], dtype=float)
                        diagmaker[[ij for ij in range(indleft, indright + 1)]] = 1.0
                        #templist = np.diag(diagmaker) * ThisDlist
                        templist = sparse.diags(diagmaker, 0) * ThisDlist
                        ThisDlistshader = ThisDlistshader + templist
                        #ThisDlistshader[[ij for ij in range(indleft, indright + 1)],:] = 1.0
                #GIntermediate = np.multiply(ThisDlist, ThisDlistshader).transpose()
                #gradhelper[i,:] = GIntermediate.sum(axis=1)
                print(gradhelper.shape)
                gradhelper[:,i] = ThisDlistshader.transpose().sum(axis=1)

        self.GradientIntermediate = gradhelper

    def calcGradientandObjValue(self):
        oDoseObj = self.currentDose - quadHelperThresh
        oDoseObjCl = (oDoseObj > 0) * oDoseObj
        oDoseObj = (oDoseObj > 0) * oDoseObj
        oDoseObj = oDoseObj * oDoseObj * quadHelperOver

        uDoseObj = quadHelperThresh - self.currentDose
        uDoseObjCl = (uDoseObj > 0) * uDoseObj
        uDoseObj = (uDoseObj > 0) * uDoseObj
        uDoseObj = uDoseObj * uDoseObj * quadHelperUnder
        self.objectiveValue = sum(oDoseObj + uDoseObj)

        oDoseObjGl = 2 * oDoseObjCl * quadHelperOver
        uDoseObjGl = 2 * uDoseObjCl * quadHelperUnder

        # This is the gradient of dF / dZ. Dimension is numvoxels
        self.mygradient = 2 * (oDoseObjGl - uDoseObjGl)
        # This is the gradient of dF / dk. Dimension is num Apertures
        self.scipygradient = (np.asmatrix(self.mygradient) * self.GradientIntermediate).transpose()

    # default constructor
    def __init__(self):
        self.numX = 0

########## END OF CLASS DECLARATION ###########################################

catemp = []
gatemp = [] 
for thisline in mylines:
    a, b = thisline.split('\t')
    if (int(float(a)) % 10 == 0):
        if(int(float(b)) % 10 == 0):
            catemp.append(a)
            gatemp.append(b)

# First of all make sure that I can read the data

# In the data directory with the *VOILIST.mat files, this opens up
# each structure file and reads in the structure names and sizes in
# voxels

start = time.time()
data = vmat_class()

data.outputDirectory = outputfolder # given by the user in the first lines of *.pydoc
data.dataDirectory = readfolder

# Function definitions
####################################################################
def readctvoxelinfo():
    # This function returns a dictionary with the dimension in voxel
    # units for x,y,z axis
    
    lines = [line.rstrip('\n') for line in open(readfolder + 'CTVOXEL_INFO.txt')]
    tempocoor = []
    for i in range(0,3):
        tempocoor.append(int(lines[i].rsplit(None, 1)[-1]))
    coordims = dict(x=tempocoor[0],y=tempocoor[1],z=tempocoor[2])
    return(coordims)
#########################################•••••••••###########################
oldfolder = os.getcwd()
os.chdir(readfolder)
allFiles = glob.glob("*VOILIST.mat")
allBeamInfos = glob.glob("*Couch0_BEAMINFO.mat")
allNames = sorted(allFiles) #Make sure it's sorted because it was not.
allBeamInfoNames = sorted(allBeamInfos)
numStructs = len(allFiles)

# This is "big voxel space" where some voxels may receive no dose or
# have no structure assigned
vdims = readctvoxelinfo()
numVoxels = vdims['x'] * vdims['y'] * vdims['z']

Vorg = []
bigZ = np.zeros(numVoxels, dtype=int)

# Vorg is a list of the structure voxels in big voxel space
for s in range(0, numStructs):
    Vorg.append(sio.loadmat(allNames[s])['v']-1) # correct 1 position mat2Py.
    bigZ[Vorg[s]] = 1.0

# nVox is "small voxel space", with only the voxels that have
# structures assigned (basically non-air/couch voxels)
nVox = sum(bigZ);

# voxelAssignment provides the mapping from small voxel space to big
# voxel space.
data.voxelAssignment = np.empty(nVox.astype(np.int64))
data.voxelAssignment[:] = np.NAN

counter = 0
for i in range(0, numVoxels):
    if(bigZ[i] > 0):
        # If big space voxel is nonzero, save to small vxl space
        data.voxelAssignment[counter] = i
        counter+=1
print('mapping from small voxel space to big voxel space done')

# originalVoxels is the mapping from big voxel space to small voxel
# space

# It is VERY important to initialize originalVoxels with NAN in this case.
# Or you can make an error since 0 is a valid position in python.
originalVoxels = np.empty(numVoxels); originalVoxels[:] = np.NAN
for i in range(0, nVox.astype(np.int64)):
    originalVoxels[data.voxelAssignment[i].astype(np.int64)] = i

## Read in structures . CHANGE THIS. Reading from txt file != good!!
lines = [myline.split('\t') for myline in [line.rstrip('\n') for line in open(structurefile)]]
## Collapse the above expression to a flat list
invec = [item for sublist in lines for item in sublist]
## Assignation of different values
data.numstructs = int(invec[2])
data.numtargets = int(invec[3])
data.numoars = int(invec[4])
# Structure map OARs vs. TARGETs
data.regionIndices = invec[5:(5+data.numstructs)]
data.targets = invec[(5+data.numstructs):(5+2*data.numstructs)]
data.oars = invec[(5+2*data.numstructs):(5+3*(data.numstructs))]
print('Finished reading structures')

maskValueFull = np.zeros(nVox.astype(np.int64))
maskValueSingle = np.zeros(nVox.astype(np.int64))
# this priority is the order of priority for assigning a single structure per
# voxel (from least to most important)

# CAREFUL!!!! masking value gets indices that agree with Troy's matlab implemen
# tation. My reasoning is that I want to be compatible with his code down the
# road. minimum maskin value will be 1 (one).
for i in range(0, numStructs):
    s = priority[i]
    # generates mask values (the integer that we decompose to get structure
    # assignment). for single it just overrides with the more important
    # structure
    maskValueFull[originalVoxels[Vorg[s]].astype(int)] = maskValueFull[originalVoxels[Vorg[s]].astype(int)]+2**(s)
    maskValueSingle[originalVoxels[Vorg[s]].astype(int)] = 2**(s)
    # print('s: ' + str(s) + ', mValue:' + str(maskValueFull[111001]))

print('masking value single from ' + str(min(maskValueSingle)) + ' to ' + str(max(maskValueSingle)))

# Reverse the list for the full mask value. norepeat contains all original values
# and values will be removed as they get assigned. This is to achieve precedence
# TROY!. My regions are not organized alphabetically but in inverse order of
# priority. So they won't match unless you look for the right one.
priority.reverse()
norepeat = np.unique(originalVoxels[np.invert(np.isnan(originalVoxels))])
for s in priority:
    # initialize regions
    istarget = str(s) in data.targets
    tempindicesfull = originalVoxels[Vorg[s]].astype(int) # In small voxels space
    tempindices = np.intersect1d(tempindicesfull, norepeat)
    print("initialize region " + str(s) + ', full indices: ' + str(len(tempindicesfull)) + ', and single indices: ' + str(len(tempindices)))
    data.regions.append(region(s, tempindices, tempindicesfull, istarget))
    # update the norepeat vector by removing the newly assigned indices
    norepeat = np.setdiff1d(norepeat, tempindices)

print('finished assigning voxels to regions. Region objects read')
 
# Read in mask values into structure data
data.maskValue = maskValueSingle
data.fullMaskValue = maskValueFull
print('Masking has been calculated')

gastart = 0 ;
gaend = 356;
gastep = 60;
castart = 0;
caend = 0;
castep = 0;
ga=[];
ca=[];

## Treatment of BEAMINFO data

os.chdir(readfolderD)
for g in range(gastart, gaend, gastep):
    fname = 'Gantry' + str(g) + '_Couch' + str(0) + '_D.mat'
    bletfname = readfolder + 'Gantry' + str(g) + '_Couch' + str(0) + '_BEAMINFO.mat'
    if os.path.isfile(fname) and os.path.isfile(bletfname):
        ga.append(g)
        ca.append(0)

print('There is enough data for ' + str(len(ga)) + ' beam angles\n')

# build new sparse matrices

# This code translates the sparse dose matrices from big voxel space to
# small voxel space and writes it out to a binary file to be used in the
# optimization
nBPB = np.zeros(len(ga))
# nDIJSPB is the number of nonzeros in the Dmatrix for each beam
nDIJSPB = np.zeros(len(ga))

###############################################################################
## Beginning of Troy's cpp code (interpreted, not copied)

## This comes from first two lines in doseInputs txt file (troy's version)
data.numvoxels = nVox
data.numbeams = len(ga)
## Allocate memory
data.beamletsPerBeam = np.empty(data.numbeams, dtype=int)
data.dijsPerBeam = np.empty(data.numbeams, dtype=int)
data.xdirection = []
data.ydirection = []
beamletCounter = np.zeros(data.numbeams + 1)

for i in range(0, data.numbeams):
    bletfname = readfolder + 'Gantry' + str(ga[i]) + '_Couch' + str(0) + '_BEAMINFO.mat'
    # Get beamlet information
    binfoholder = sio.loadmat(bletfname)

    # Get dose information as in the cpp file
    data.beamletsPerBeam[i] = int(binfoholder['numBeamlets'])
    data.dijsPerBeam[i] =  int(binfoholder['numberNonZerosDij'])
    data.xdirection.append(binfoholder['x'][0])
    data.ydirection.append(binfoholder['y'][0])
    if 0 == i:
        data.xinter = data.xdirection[0]
        data.yinter = data.ydirection[0]
    else:
        data.xinter = np.intersect1d(data.xinter, data.xdirection[i])
        data.yinter = np.intersect1d(data.yinter, data.ydirection[i])
## After reading the beaminfo information. Read CUT the data.

N = len(data.yinter) #N will be related to the Y axis.
M = len(data.xinter) #M will be related to the X axis.

###################################################
## Initial intensities are allocated a value of zero.
data.currentIntensities = np.zeros(data.numbeams, dtype = float)

# Generating dose matrix dimensions
data.numX = sum(data.beamletsPerBeam)
data.totaldijs = sum(data.dijsPerBeam)
# Allocate structure for full Dmat file
data.Dmat = sparse.csr_matrix((data.numX, data.numvoxels), dtype=float)

# Work with the D matrices for each beam angle
overallDijsCounter = 0
Dlist = []
for i in range(0, data.numbeams):
    fname = 'Gantry' + str(ga[i]) + '_Couch' + str(0) + '_D.mat'
    print('Processing matrix from gantry & couch angle: ' + fname)
    # extract voxel, beamlet indices and dose values
    D = sio.loadmat(fname)['D']
    # write out bixel sorted binary file
    [b,j,d] = sparse.find(D)
    newb = originalVoxels[b]
    
    # write out voxel sorted binary file
    [jt,bt,dt] = sparse.find(D.transpose())
    newbt = originalVoxels[bt]
    Dlittle = sparse.csr_matrix((dt, (jt, newbt)), shape = (data.numX, data.numvoxels), dtype=float)
    # For each matrix D, store its values in a list
    Dlist.append(Dlittle)

print('Finished reading D matrices')

### Here I begin the matrix cut

for i in range(0, data.numbeams):
    # ininter will contain the elements that belong in the intersection of all beamlets
    ininter = []
    for j in range(0, len(data.xdirection[i])):
        if (data.xdirection[i][j] in data.xinter and data.ydirection[i][j] in data.yinter):
            ininter.append(j)

    # Once I have ininter I will cut all the elements that are
    data.xdirection[i] = data.xdirection[i][ininter]
    data.ydirection[i] = data.ydirection[i][ininter]

    Dlist[i] = Dlist[i][ininter,:]
    data.beamletsPerBeam[i] = len(ininter)
    beamletCounter[i+1] = beamletCounter[i] + data.beamletsPerBeam[i]

data.numX = sum(data.beamletsPerBeam)

#### MATRIX CUT DONE Here all matrices are working with the same limits

## Read in the objective file:
lines = [myline.split('\t') for myline in [line.rstrip('\n') for line in open(objfile)]]
## Collapse the above expression to a flat list
data.functionData = [item for sublist in lines for item in sublist]
data.objectiveInputFiles = objfile
print("Finished reading objective file:\n" + objfile)

## Read in the constraint file:
#####NOTHING TO DO #############

# Reading algorithm Settings
data.algOptions = [myline.split('\t') for myline in [line.rstrip('\n') for line in open(algfile)]]
print("Finished reading algorithm inputs file:\n" + algfile)

# resize dose and beamlet vectors
data.currentDose = np.zeros(data.numvoxels)
####################################
### FINISHED READING EVERYTHING ####
####################################

## Work with function data.
data.functionData = np.array([float(i) for i in data.functionData[3:len(data.functionData)]]).reshape(3,data.numstructs)
# I have to reorder the right region since my order is not alphabetical
data.functionData = data.functionData[:,priority]
functionData = data.functionData
for s in range(0, data.numstructs):
    if(data.regions[s].sizeInVoxels > 0):
        functionData[1,s] = functionData[1,s] * 1 / data.regions[s].sizeInVoxels
        functionData[2,s] = functionData[2,s] * 1 / data.regions[s].sizeInVoxels

# initialize helper variables
quadHelperThresh = np.zeros(data.numvoxels)
quadHelperOver = np.zeros(data.numvoxels)
quadHelperUnder = np.zeros(data.numvoxels)
quadHelperAlphaBetas = np.zeros(data.numvoxels)
uDose = np.zeros(data.numvoxels)
oDose = np.zeros(data.numvoxels)

# build for each voxel
for s in range(0, data.numstructs):
    for j in range(0, data.regions[s].sizeInVoxels):
        quadHelperThresh[int(data.regions[s].indices[j])] = functionData[0][s]
        quadHelperOver[int(data.regions[s].indices[j])] = functionData[1][s]
        quadHelperUnder[int(data.regions[s].indices[j])] = functionData[2][s]

def calcObjGrad(x, user_data = None):
    data.currentIntensities = x
    data.calcDose()
    data.calcGradientandObjValue()
    print("data.scipygradient from: ", data.scipygradient)
    return(data.objectiveValue, data.scipygradient)

def eval_g(x, user_data= None):
           return array([], float_)

def eval_jac_g(x, flag, user_data = None):
    if flag:
        return ([], [])
    else:
        return array([])

def PPsubroutine(C, C2, C3, b, angdistancem, angdistancep, vmax, speedlim, predec, succ, N, M, index):
    # C, C2, C3 are constants in the penalization function
    # angdistancem = $\delta_{c^-c}$
    # angdistancep = $\delta_{cc^+}$
    # vmax = maximum leaf speed
    # speedlim = s
    # predec = predecesor index, either an index or an empty list
    # succ = succesor index, either an index or an empty list
    # lcm = vector of left limits in the previous aperture
    # lcp = vector of left limits in the next aperture
    # rcm = vector of right limits in the previous aperture
    # rcp = vector of right limits in the previous aperture
    # N = Number of beamlets per row
    # M = Number of rows in an aperture
    # index = index location in the set of apertures that I have saved.
    
    networkNodes = []
    # Start with arcs that go from the source to level m = 1
    # Create source node
    networkNodes.append([-1, 0, 0, 0, 0]) # m, l, r, distance, index of predecesor
    posBeginningOfRow = 1
    D = Dlist[index]

    # vmaxm and vmaxp describe the speeds that are possible for the leaves from the predecessor and to the successor
    vmaxm = vmax
    vmaxp = vmax
    # Arranging the predecessors and the succesors.
    #Predecessor left and right indices
    if type(predec) is list:
        lcm = [0] * M
        rcm = [N] * M
        # If there is no predecessor is as if the pred. speed was infinite
        vmaxm = float("inf")
    else:
        lcm = data.llist[predec]
        rcm = data.rlist[predec]

    #Succesors left and right indices
    if type(succ) is list:
        lcp = [0] * M
        rcp = [N] * M
        # If there is no successor is as if the succ. speed was infinite.
        vmaxp = float("inf")
    else:
        lcp = data.llist[succ]
        rcp = data.rlist[succ]

    # Find geographical location of the first row.
    geolocX = data.xinter[0]
    # Find all possible locations of beamlets in this row according to geographical location
    indys = np.where(geolocX == data.xdirection[index])
    ys = data.ydirection[index][indys]
    validbeamlets = np.in1d(data.yinter, ys)
    validbeamlets = np.array(range(0, len(data.yinter)))[validbeamlets]
    # Keep the location of the most leaf
    leftmostleaf = len(ys) # Position in python position(-1) of the leftmost leaf
    nodesinpreviouslevel = 0
    oldflag = nodesinpreviouslevel
    # First handle the calculations for the first row

    for l in range(math.ceil(max(min(validbeamlets) - 1, lcm[0] - vmaxm * angdistancem/speedlim, lcp[0] - vmaxp * angdistancep / speedlim)), math.floor(min(max(validbeamlets), lcm[0] + vmaxm * angdistancem / speedlim, lcp[0] + vmaxp * angdistancep / speedlim))):
        for r in range(math.ceil(max(l + 1, rcm[0] - vmaxm * angdistancem/speedlim, rcp[0] - vmaxp * angdistancep / speedlim)), math.floor(min(max(validbeamlets)+1, rcm[0] + vmaxm * angdistancem / speedlim, rcp[0] + vmaxp * angdistancep / speedlim))):

            # First I have to make sure to add the beamlets that I am interested in
            if(l + 1 <= r -1): # prints r numbers starting from l + 1. So range(3,4) = 3
                Dose = -sum( D[[i for i in range(l+1, r)],:] * data.mygradient)
                weight = C * ( C2 * (r - l) - C3 * b * (r - l)) - Dose
            else:
                weight = 0.0
            # Create node (1,l,r) in array of existing nodes and update the counter
            networkNodes.append([1, l, r, weight, 0])
            nodesinpreviouslevel = nodesinpreviouslevel + 1

    posBeginningOfRow = posBeginningOfRow + nodesinpreviouslevel
    mystart = time.time()
    
    # Then handle the calculations for the m rows. Nodes that are neither source nor sink.
    for m in range(2,M):
        # Show time taken per row
        print("Now on row", m)
        myend   =  time.time()
        print(myend - mystart)
        mystart = myend
        # Find geographical location of this row.
        geolocX = data.xinter[m-1]
        # Find all possible locations of beamlets in this row according to geography
        indys = np.where(geolocX == data.xdirection[index])
        ys = data.ydirection[index][indys]
        validbeamlets = np.in1d(data.yinter, ys)
        validbeamlets = np.array(range(0, len(data.yinter)))[validbeamlets]
        oldflag = nodesinpreviouslevel
        nodesinpreviouslevel = 0

        # And now process normally checking against valid beamlets
        for l in range(math.ceil(max(min(validbeamlets)-1, lcm[m] - vmaxm * angdistancem/speedlim, lcp[m] - vmaxp * angdistancep / speedlim)), math.floor(min(max(validbeamlets), lcm[m] + vmaxm * angdistancem / speedlim, lcp[m] + vmaxp * angdistancep / speedlim))):
            for r in range(math.ceil(max(l + 1, rcm[m] - vmaxm * angdistancem/speedlim, rcp[m] - vmaxp * angdistancep / speedlim)), math.floor(min(max(validbeamlets) + 1, rcm[m] + vmaxm * angdistancem / speedlim, rcp[m] + vmaxp * angdistancep / speedlim))):
                # Create node (m, l, r) and update the level counter
                networkNodes.append([m, l, r, float("inf"), float("inf")])
                nodesinpreviouslevel = nodesinpreviouslevel + 1
                thisnode = len(networkNodes) - 1
                lmlimit = leftmostleaf
                rmlimit = (r - l) + leftmostleaf
                if(lmlimit + 1 <= rmlimit - 1):
                    Dose = - sum(D[[i for i in range(lmlimit + 1, rmlimit)],:] * data.mygradient)
                    C3simplifier = C3 * b * (r - l)
                else:
                    Dose = 0.0
                    C3simplifier = 0
                for mynode in (range(posBeginningOfRow - oldflag, posBeginningOfRow)):
                    # Create arc from (m-1, l, r) to (m, l, r). And assign weight
                    lambdaletter = math.fabs(networkNodes[mynode][1] - l) + math.fabs(networkNodes[mynode][2] - r) - 2 * max(0, networkNodes[mynode][1] - r) - 2 * max(0, l - math.fabs(networkNodes[mynode][2]))
                    weight = C * (C2 * lambdaletter - C3simplifier) - Dose
                    #print("weight is", weight)
                    if(networkNodes[mynode][3] + weight < networkNodes[thisnode][3]):
                        networkNodes[thisnode][3] = networkNodes[mynode][3] + weight
                        # And next we look for the minimum distance.
                        networkNodes[thisnode][4] = mynode
        posBeginningOfRow = nodesinpreviouslevel + posBeginningOfRow # This is the total number of network nodes
        # Keep the location of the leftmost leaf
        leftmostleaf = len(ys) + leftmostleaf

    print("And last. Add the arcs to the sink")
    networkNodes.append([M, float("inf"), float("inf"), float("inf"), float("inf")])
    posBeginningOfRow = posBeginningOfRow + 1 # Notice that this position right now falls outside the array! Here only for illustration.
    thisnode = len(networkNodes) - 1
    for mynode in (range(posBeginningOfRow - nodesinpreviouslevel, posBeginningOfRow)):
        weight = C * ( C2 * (networkNodes[mynode][2] - networkNodes[mynode][1] ) )
        if(networkNodes[mynode][3] + weight <= networkNodes[thisnode][3]):
            networkNodes[thisnode][3] = networkNodes[mynode][3] + weight
            networkNodes[thisnode][4] = mynode
            p = networkNodes[thisnode][3]
    
    # return set of left and right limits
    thenode = len(networkNodes) - 1
    l = []
    r = []
    while(1):
        # Find the predecessor data
        print(thenode)
        l.append(networkNodes[thenode][1])
        r.append(networkNodes[thenode][2])
        thenode = networkNodes[thenode][4]
        if(0 == thenode): # If at the origin then break.
            break
    l.reverse()
    r.reverse()
    print(p)
    return(p, l, r)

def PricingProblem(C, C2, C3, b, angdistancem, angdistancep, vmax, speedlim, N, M):
    lall = []
    rall = []
    pstar = float("inf")
    i = 0
    # This is just for debugging
    #for index in data.notinC:
    # Wilmer. Fix this, this is only going to index 0 for debugging purposes
    #for index in data.notinC:
    for index in [0]:
        print("analysing index" , index)
        # Find the succesor and predecessor of this particular element
        try:
            succs = [i for i in range(0, data.caligraphicC) if i > index]
        except:
            succs = []
        try:
            predecs = [i for i in range(0, data.caligraphicC) if i < index]
        except:
            predecs = []

        # If there are no predecessors or succesors just return an empty list. If there ARE, then return the indices
        if 0 == len(succs):
            succ = []
        else:
            succ = min(succs)
        if 0 == len(predecs):
            predec = []
        else:
            predec = max(predecs)

        p, l, r = PPsubroutine(C, C2, C3, b, angdistancem, angdistancep, vmax, speedlim, predec, succ, N, M, index)
        lall.append(l)
        rall.append(r)
        # The next "if" will be entered at least once.
        if p < pstar:
            bestAperture = index
            pstar = p
            besti = i
        i = i + 1
    print("the best aperture was: ", bestAperture)
    return(pstar, lall[besti], rall[besti], bestAperture)

def solveRMC():
    ## IPOPT SOLUTION
    start = time.time()
    numbe = len(data.caligraphicC)
    nvar = numbe
    xl = np.zeros(numbe)
    xu = 2e19*np.ones(numbe)
    m = 0
    gl = np.zeros(1)
    gu = 2e19*np.ones(1)
    g_L = np.array([], dtype=float)
    g_U = np.array([], dtype=float)
    nnzj = 0
    nnzh = int(numbe * (numbe + 1) / 2)

    calcObjGrad(data.currentIntensities)
    print("type of x0:", type(data.currentIntensities))
    print(data.currentIntensities.shape)
    # Create the boundaries making sure that the only free variables are the ones with perfectly defined apertures.
    boundschoice = []
    for thisindex in range(0, data.numbeams):
        if thisindex in data.notinC:
            boundschoice.append((0, 0))
        else:
            boundschoice.append((0, None))
    print(boundschoice)
    res = minimize(calcObjGrad, data.currentIntensities, method='L-BFGS-B', jac = True, bounds = boundschoice, options={'ftol':1e-6, 'disp':5,'maxiter':1000})

    print('solved in ' + str(time.time() - start) + ' seconds')
    
def colGen():
    # User defined data
    C = 1.0
    C2 = 1.0
    C3 = 1.0
    angdistancem = 60
    angdistancep = 60
    vmax = 2.0
    speedlim = 3.0


    # Assign the most open apertures as initial apertures. They will not have any energy applied to them.
    for i in range(0, data.numbeams):
        data.llist.append([-1] * len(data.xinter))
        data.rlist.append([len(data.yinter) + 1] * len(data.xinter))

    #Step 0 on Fei's paper. Set C = empty and zbar = 0. The gradient of numbeams dimensions generated here will not
    # be used, and therefore is nothing to worry about.
    data.calcDose()

    # At the beginning no apertures are selected, and those who are not selected are all in notinC
    data.caligraphicC = []
    data.notinC = [i for i in range(0, len(Dlist))]

    pstar = -float("inf")
    while(pstar < 0):
        # Step 1 on Fei's paper. Use the information on the current treatment plan to formulate and solve an instance of the PP
        data.calcDose()
        data.calcGradientandObjValue()
        #p, lm, rm =  PPsubroutine(C, C2, C3, 0.5, angdistancem, angdistancep, vmax, speedlim, 4, [], N, M, 5)
        p, lm, rm, bestAperture = PricingProblem(C, C2, C3, 0.5, angdistancem, angdistancep, vmax, speedlim, N, M)

        # Step 2. If the optimal value of the PP is nonnegative**, go to step 5. Otherwise, denote the optimal solution to the
        # PP by c and Ac and replace caligraphic C and A = Abar, k \in caligraphicC
        if p >= 0:
            break
        else:
            data.caligraphicC.append(bestAperture)
            data.caligraphicC.sort()
            print("best Aperture is:", bestAperture)
            data.notinC.remove(bestAperture)
            data.notinC.sort()
            # Solve the instance of the RMP associated with caligraphicC and Ak = A_k^bar, k \in caligraphicC
            data.llist[bestAperture] = lm
            data.rlist[bestAperture] = rm
            solveRMC()

            #Step 5 on Fei's paper. If necessary complete the treatment plan by identifying feasible apertures at control points c
            #notinC and denote the final set of fluence rates by yk


print('Preparation time took: ' + str(time.time()-start) + ' seconds')

colGen()

# PYTHON scipy.optimize solution

# find initial location
# res = minimize(calcObjGrad, data.currentIntensities,method='L-BFGS-B', jac = True, bounds=[(0, None) for i in range(0, len(data.currentIntensities))], options={'ftol':1e-3,'disp':5,'maxiter':1000,'gtol':1e-3})
# Print results
