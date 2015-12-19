
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
import pyipopt
import numpy as np
import scipy.io as sio
from scipy import sparse
from scipy.optimize import minimize
import time
import math

# First of all make sure that I can read the data

# In the data directory with the *VOILIST.mat files, this opens up
# each structure file and reads in the structure names and sizes in
# voxels

start = time.time()

readfolder = '/home/wilmer/Documents/Troy_BU/Data/DataProject/HN/'
readfolderD = readfolder + 'Dij/'
outputfolder = '/home/wilmer/Dropbox/Research/VMAT/output/'
degreesep = 60 # How many degrees in between separating neighbor beams.
objfile = '/home/wilmer/Dropbox/IpOptSolver/TestData/HNdata/objectives/obj1.txt'
structurefile = '/home/wilmer/Dropbox/IpOptSolver/TestData/HNdata/structureInputs.txt'
algfile = '/home/wilmer/Dropbox/IpOptSolver/TestData/HNdata/algInputsWilmer.txt'
mm3voxels = '/home/wilmer/Documents/Troy_BU/Data/DataProject/HN/hn3mmvoxels.mat'
# The 1 is subtracted at read time so the user doesn't have to do it everytime
priority = [7, 24, 25, 23, 22, 21, 20, 16, 15, 14, 13, 12, 10, 11, 9, 4, 3, 1, 2, 17, 18, 19, 5, 6, 8]
priority = (np.array(priority)-1).tolist()
mylines = [line.rstrip('\n') for line in open('/home/wilmer/Dropbox/Research/VMAT/VMATwPenCode/beamAngles.txt')]

catemp = []
gatemp = [] 
for thisline in mylines:
    a, b = thisline.split('\t')
    if (int(float(a)) % 10 == 0):
        if(int(float(b)) % 10 == 0):
            catemp.append(a)
            gatemp.append(b)

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
    beamNumPerBeam = [] # beam index per beam (for reading in individual beams)
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

    # big D matrix
    Dmat = sparse.csr_matrix((1,1), dtype=float) # sparse Dij matrix
    
    # varios folders
    outputDirectory = outputfolder # given by the user in the first lines of *.py
    dataDirectory = readfolder

    # dose variables
    currentDose = [] # dose variable
    currentIntensities = []
    
    caligraphicC = []
    # this is the intersection of all beamlets that I am dealing with
    xinter = []
    yinter = []
    xdirection = []
    ydirection = []
    
    # data class function
    def calcDose(self, newIntensities):
        self.currentIntensities = newIntensities
        # self.currentDose = self.Dmat.transpose() * newIntensities
        self.currentDose = np.zeros()
        for i in self.caligraphicC:
            self.currentDose += Dlist[i] * np.repeat(self.currentIntensities[i], Dlist[i].shape[1], axis = 0)

    # default constructor
    def __init__(self):
        self.numX = 0

########## END OF CLASS DECLARATION ###########################################
data = vmat_class()

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
####################################################################
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

## Read in structures WILMER. CHANGE THIS. Reading from txt file != good!!
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

# nBPB is the  number of beamlets per beam
nBPB = np.zeros(len(ga))
# nDIJSPB is the number of nonzeros in the Dmatrix for each beam
nDIJSPB = np.zeros(len(ga))

###############################################################################
## Beginning of Troy's cpp code (interpreted, not copied)

## This comes from first two lines in doseInputs txt file (troy's version)
data.numvoxels = nVox
data.numbeams = len(ga)
## Allocate memory
data.beamNumPerBeam = np.empty(data.numbeams, dtype=int)
data.beamletsPerBeam = np.empty(data.numbeams, dtype=int)
data.dijsPerBeam = np.empty(data.numbeams, dtype=int)
data.xdirection = []
data.ydirection =[]
beamletCounter = np.zeros(data.numbeams + 1)

for i in range(0, data.numbeams):
    bletfname = readfolder + 'Gantry' + str(ga[i]) + '_Couch' + str(0) + '_BEAMINFO.mat'
    # Get beamlet information
    binfoholder = sio.loadmat(bletfname)

    # Get dose information as in the cpp file
    # data.beamNumPerBeam[i] = 10 * int(i) # WILMER. Find out why!!!
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

###################################################


## Initial intensities are allocated a value of zero.
data.currentIntensities = np.zeros(data.numbeams, dtype=float)

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
    # For each matrix D, store its values in a list
    Dlist.append(D)

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

    Dlist[i] = Dlist[i][:,ininter]
    data.beamletsPerBeam[i] = len(ininter)
    beamletCounter[i+1] = beamletCounter[i] + data.beamletsPerBeam[i]

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

def PPsubroutine(C, C2, C3, angdistancem, angdistancep, vmax, speedlim, lcm, lcp, rcm, rcp, N, M):
    # C, C2, C3 are constants in the penalization function
    # angdistancem = $\delta_{c^-c}$
    # angdistancep = $\delta_{cc^+}$
    # vmax = maximum leaf speed
    # speedlim = s
    # lcm = vector of left limits in the previous aperture
    # lcp = vector of left limits in the next aperture
    # rcm = vector of left limits in the previous aperture
    # rcm = vector of right limits in the previous aperture
    # N = Number of beamlets per row
    # M = Number of rows in an aperture
    
    networkNodes = []
    networkArcs = [] # contains pair of elements that it connects and weight
    flagposition = 0
    nodesinpreviouslevel = 0
    # Start with arcs that go from the source to level m = 1
    # Create source node
    networkNodes.append([0, 0, 0, 0, 0]) # m, l, r, distance, predecesor
    boundaries = []
    minweight = math.inf
    for l in range(math.ceil(max(0, lcm[0] - vmax * angdistancem/speedlim, lcp[0] - vmax * angdistancep / speedlim)), math.floor(min(N, lcm[0] + vmax * angdistancem / speedlim, lcp[0] + vmax * angdistancep / speedlim))):
        for r in range(math.ceil(max(l + 1, rcm[0] - vmax * angdistancem/speedlim, rcp[0] - vmax * angdistancep / speedlim)), math.floor(min(N+1, rcm[0] + vmax * angdistancem / speedlim, rcp[0] + vmax * angdistancep / speedlim))):
            # Create arc from source to (1, l, r) and assign a vector weight to it.
            # First I have to make sure to add the beamlets that I am interested in
            Dose = sum(D[range(l,r),:] * data.mygradient)
            weight = C * (C2 * (r - l) - C3 * b * (r - l) - Dose)
            networkArcs.append([1, len(networkNodes), weight])
            # Create node (1,l,r) in array of existing nodes
            networkNodes.append([1, l, r, weight, 0])
            # Find the least element in the weight list.
            if (weight < minweight):
                minweight = weight
                bl = l
                br = r
            boundaries.append([bl, br])
            flagposition = flagposition + 1
            nodesinpreviouslevel = nodesinpreviouslevel + 1

    for m in range(2,M):
        flagnewlevel = 0
        for l in range(math.ceil(max(0, lcm[0] - vmax * angdistancem/speedlim, lcp[0] - vmax * angdistancep / speedlim)), math.floor(min(N, lcm[0] + vmax * angdistancem / speedlim, lcp[0] + vmax * angdistancep / speedlim))):
            for r in range(math.ceil(max(l + 1, rcm[0] - vmax * angdistancem/speedlim, rcp[0] - vmax * angdistancep / speedlim)), math.floor(min(N+1, rcm[0] + vmax * angdistancem / speedlim, rcp[0] + vmax * angdistancep / speedlim))):
                flagnewlevel = flagnewlevel + 1
                # Create node (m, l, r)
                networkNodes.append(m, l, r, math.inf, 0)
                thisnode = len(networkNodes)
                for mynode in (range(flagposition - nodesinpreviouslevel, flagposition)):
                    # Create arc from (m-1, l, r) to (m, l, r). And assign weight
                    lambdaletter = math.fabs(networkNodes[mynode][1] - l) + math.fabs(networkNodes[mynode][2] - r) - 2 * max(0, networkNodes[mynode][1] - r) - 2 * max(0, l - math.fabs(networkNodes[mynode][2]))
                    lmlimit = l + ((m - 1) * N)                    
                    rm = r + ((m - 1) * N)
                    Dose = sum(D[range(lmlimit, rm),:] * data.mygradient)
                    weight = C(C2 * lambdaletter - C3 * b * (rm - lmlimit)) - sum(D[range(l,r),:] * data.nablaF)
                    if(networkNodes[mynode][3] + weight < networkNodes[thisnode]):
                        networkNodes[thisnode][3] = networkNodes[mynode][3] + weight
                        # And next we look for the minimum distance.
                        networkNodes[thisnode][4] = mynode
        flagpositiion = flagnewlevel + flagposition
        nodesinpreviouslevel = flagnewlevel
    
    # And last. Add the arcs to the sink
    networkNodes.append([M + 1, 0, 0, math.Inf, 0])
    thisnode = len(networkNodes)
    for mynode in (range(flagposition - nodesinpreviouslevel, flagposition)):
        weight = C * ( C2 * (r - l))
        if(networkNodes[mynode][3] + weight <= networkNodes[thisnode]):
            networkNodes[mynode][3] = networkNodes[mynode][3] + weight
            networkNodes[mynode][4] = mynode
            p = networkNodes[mynode][3]

    # return set of left and right limits
    thenode = len(networkNodes)
    l = []
    r = []
    while(1):
        # Find the predecessor
        thenode = networkNodes[thenode][4]
        l.append(networkNodes[thenode][1])
        r.append(networkNodes[thenode][2])
        
    return(p, reversed(l), reversed(r))

def solveRMC():
    
    data.numX = sum(data.beamletsPerBeam)
    data.Dmat = sparse.csr_matrix((data.numX, data.numvoxels), dtype=float)
    for i in range(0, data.numbeams):
        [jt,bt,dt] = sparse.find(Dlist[i].transpose())
        newbt = originalVoxels[bt]
        # Notice here that python is smart enough to subtract 1 from matlab's mat
        # files (where the index goes). This was confirmed by Wilmer on 10/19/2015
        tempsparse=sparse.csr_matrix((dt,(jt + beamletCounter[i], newbt)),
                                    shape=(data.numX, data.numvoxels), dtype=float)
        data.Dmat = data.Dmat + tempsparse
    
def colGen():
    data.caligraphicC = []
    notinC = range(0, len(Dlist))
    zlist = np.zeros(len(Dlist))
    iflag = 0
    pstar = math.inf
    for i in notinC:
        p, l, r = PPsubroutine(C, C2, C3, angdistancem, angdistancep, vmax, speedlim, lcm, lcp, rcm, rcp, N, M)
        if p < pstar:
            pstar = p
            iflag = i
        if pstar > 0:
            data.caligraphicC.append(iflag)
            solveRMC

Clist = []
Alist = range(1, 360 / degreesep)
z = 0
for aperture in set(Alist) - set(Clist):
    
def evaluateFG(x, user_data=None):
    data.calcDose(x)
    
    oDoseObj = data.currentDose - quadHelperThresh
    oDoseObjCl = (oDoseObj > 0) * oDoseObj
    oDoseObj = (oDoseObj > 0) * oDoseObj
    oDoseObj = oDoseObj * oDoseObj * quadHelperOver
    
    uDoseObj = quadHelperThresh - data.currentDose
    uDoseObjCl = (uDoseObj > 0) * uDoseObj
    uDoseObj = (uDoseObj > 0) * uDoseObj
    uDoseObj = uDoseObj * uDoseObj * quadHelperUnder
    objectiveValue = sum(oDoseObj + uDoseObj)
    
    oDoseObjGl = 2 * oDoseObjCl * quadHelperOver
    uDoseObjGl = 2 * uDoseObjCl * quadHelperUnder
    # Wilmer. Is this right?
    data.mygradient = 2 * (oDoseObjGl - uDoseObjGl)
    
    return(objectiveValue)

def evaluateFunction(x, user_data= None):
    data.calcDose(x)
    oDoseObj = data.currentDose - quadHelperThresh
    oDoseObj = (oDoseObj > 0) * oDoseObj
    oDoseObj = oDoseObj * oDoseObj * quadHelperOver
    uDoseObj = quadHelperThresh - data.currentDose
    uDoseObj = (uDoseObj > 0) * uDoseObj
    uDoseObj = uDoseObj * uDoseObj * quadHelperUnder
    objectiveValue = sum(oDoseObj + uDoseObj)
    return(objectiveValue)

def evaluateGradient(x, user_data= None):
    data.calcDose(x)
    oDoseObj = data.currentDose - quadHelperThresh
    uDoseObj = quadHelperThresh - data.currentDose
    oDoseObjCl = (oDoseObj > 0) * oDoseObj
    uDoseObjCl = (uDoseObj > 0) * uDoseObj
    oDoseObjGl = oDoseObjCl * quadHelperOver
    uDoseObjGl = uDoseObjCl * quadHelperUnder
    my2gradient = data.Dmat * 2 * (oDoseObjGl - uDoseObjGl)
    return(my2gradient)

def calcObjGrad(x):
    data.calcDose(x)
    oDoseObj = data.currentDose - quadHelperThresh
    oDoseObjCl = (oDoseObj > 0) * oDoseObj
    oDoseObjGl = oDoseObjCl * oDoseObjCl * quadHelperOver
    uDoseObj = quadHelperThresh - data.currentDose
    uDoseObjCl = (uDoseObj > 0) * uDoseObj
    uDoseObjGl = uDoseObjCl * uDoseObjCl * quadHelperUnder
    objectiveValue = sum(oDoseObjGl + uDoseObjGl)
    oDoseObjGl = oDoseObjCl * quadHelperOver
    uDoseObjGl = uDoseObjCl * quadHelperUnder
    mygradient = data.Dmat * 2 * (oDoseObjGl - uDoseObjGl)
    return(objectiveValue, mygradient)

def evaluateHessian(x, lagrange, obj_factor, flag, user_data = None):
    if flag:
        # Build helper array
        data.calcDose(x)
        values = np.zeros(int(data.numX * (data.numX + 1) / 2), float_)
        #hessian = sparse.csr_matrix((data.numX, data.numX))
        quadHelperAlphaBetas = (data.currentDose < quadHelperThresh) * 2 * quadHelperUnder
        quadHelperAlphaBetas += (data.currentDose >= quadHelperThresh) * 2 * quadHelperOver
        # generate Hessian using matrix multiplication
        abDmat = obj_factor * data.Dmat *sparse.diags(quadHelperAlphaBetas, 0)* data.Dmat.transpose()
        
        hessian = abDmat.todense()
        idx = 0
        for i in range(0, data.numX):
            for j in range(0,i):
                values[idx] = hessian[i,j]
                idx += 1
    else:
        # Build helper array
        data.calcDose(x)
        values = np.zeros(int(data.numX * (data.numX + 1) / 2), float_)
        #hessian = sparse.csr_matrix((data.numX, data.numX))
        quadHelperAlphaBetas = (data.currentDose < quadHelperThresh) * 2 * quadHelperUnder
        quadHelperAlphaBetas += (data.currentDose >= quadHelperThresh) * 2 * quadHelperOver
        # generate Hessian using matrix multiplication
        abDmat = obj_factor * data.Dmat *sparse.diags(quadHelperAlphaBetas, 0)* data.Dmat.transpose()
        
        hessian = abDmat.todense()
        idx = 0
        for i in range(0, data.numX):
            for j in range(0,i):
                values[idx] = hessian[i,j]
                idx += 1
    return(values)

def eval_g(x, user_data= None):
           return array([], float_)

def eval_jac_g(x, flag, user_data = None):
    if flag:
        return ([], [])
    else:
        return array([])

print('Preparation time took: ' + str(time.time()-start) + ' seconds')
    
## IPOPT SOLUTION

start = time.time()
nvar = data.numX
xl = np.zeros(data.numX)
xu = 2e19*np.ones(data.numX)
m = 0
gl = np.zeros(1)
gu = 2e19*np.ones(1)
g_L = np.array([], dtype=float)
g_U = np.array([], dtype=float)
nnzj = 0
nnzh = int(data.numX * (data.numX + 1) / 2)

nlp = pyipopt.create(nvar, xl, xu, m, g_L, g_U, nnzj, nnzh, evaluateFunction, evaluateGradient, eval_g, eval_jac_g, evaluateHessian)
nlp.num_option('tol', 1e-5)
nlp.int_option("print_level", 5)
nlp.str_option('hessian_approximation', 'limited-memory')
nlp.str_option('mu_strategy', 'adaptive')
nlp.str_option('mu_oracle', 'probing')
nlp.str_option('linear_solver', 'ma97')
nlp.num_option('acceptable_tol', 1e-2)
nlp.int_option("acceptable_iter", 5)
nlp.num_option('acceptable_obj_change_tol', 5e-1)
mylines = [float(myline) for myline in [line.rstrip('\n') for line in open('/home/wilmer/Dropbox/IpOptSolver/currentIntensities.txt')]]
data.currentIntensities = np.array(mylines)
x, zl, zu, constraint_multipliers, obj, status = nlp.solve(data.currentIntensities)
print('solved in ' + str(time.time() - start) + ' seconds')

# PYTHON scipy.optimize solution

# find initial location
# res = minimize(calcObjGrad, data.currentIntensities,method='L-BFGS-B', jac = True, bounds=[(0, None) for i in range(0, len(data.currentIntensities))], options={'ftol':1e-3,'disp':5,'maxiter':1000,'gtol':1e-3})
# Print results

numzvectors = 1
maskValueFull = [int(i) for i in maskValueFull]
maskValueFull = np.array(maskValueFull)
print('Starting to Print Results')
for i in range(0, numzvectors):
    zvalues = data.currentDose
    maxDose = max([float(i) for i in zvalues])
    dose_resln = 0.1
    dose_ub = maxDose + 10
    bin_center = np.arange(0,dose_ub,dose_resln)
    # Generate holder matrix
    dvh_matrix = np.zeros((data.numstructs, len(bin_center)))
    lenintervalhalf = bin_center[1]/2
    # iterate through each structure
    for s in range(0,data.numstructs):
        allNames[s] = allNames[s].replace("_VOILIST.mat", "")
        #print('determining DVH for ' + pickstructs[s])
        print('dvh for structure' + str(s))
        doseHolder = zvalues[[i for i,v in enumerate(maskValueFull & 2**s) if v > 0]]
        if 0 == len(doseHolder):
            continue
        histHolder = []
        for i in bin_center:
            histHolder.append(sum(doseHolder < (i - lenintervalhalf)))
        dvhHolder = 1-(np.matrix(histHolder)/max(histHolder))
        dvh_matrix[s,] = dvhHolder

  
pylab.plot(bin_center, dvh_matrix.T, linewidth = 2.0)
plt.grid(True)

plt.xlabel('Dose Gray')
plt.ylabel('Fractional Volume')
plt.title('pyipopt 6 beams')

## special organs to plot
dvhmatrixsafe = dvh_matrix


dvh_matrix = dvhmatrixsafe
voitoplot = [18, 23, 17, 2, 8]
dvhsub = dvh_matrix[voitoplot,] 
pylab.plot(bin_center, dvhsub.T, linewidth = 2.0)

numstructs = 25
dosefile = "/home/wilmer/Dropbox/IpOptSolver/currentDose.txt"
zvalues = [ float(line.rstrip('\n'))  for line in open(dosefile)]
maskvaluefile = "/home/wilmer/Dropbox/IpOptSolver/currentMaskValue.txt"
maskValueF = [ int(line.rstrip('\n'))  for line in open(maskvaluefile)]
maskValue = np.array(maskValueF)

maxDose = max([i for i in zvalues])
dose_resln = 0.1
dose_ub = maxDose + 10
bin_center = np.arange(0, dose_ub, dose_resln)
dvh_matrix = np.zeros((numstructs, len(bin_center)))
lenintervalhalf = bin_center[1]/2
i=0

for s in range(0, numstructs):
    a = [i for i,v in enumerate(maskValue & 2**s) if v > 0]
    doseHolder = [zvalues[i] for i in a]
    if 0 == len(doseHolder):
        continue
    histHolder = []
    for i in bin_center:
        histHolder.append(sum(doseHolder < (i - lenintervalhalf)))
    dvhHolder = 1-(np.matrix(histHolder) / max(histHolder))
    dvh_matrix[s,] = dvhHolder

#pylab.plot(bin_center, alt.T, linewidth = 2.0)
dvhsub2 = dvh_matrix[voitoplot,]
pylab.plot(bin_center, dvhsub2.T, linewidth = 1.0, linestyle = '--', color = ['r', 'b', 'g', 'k', 'm'])
plt.grid(True)

plt.xlabel('Dose Gray')
plt.ylabel('Fractional Volume')
plt.title('ipopt 6 beams')

