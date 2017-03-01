#!/home/wilmer/anaconda3/bin/python

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
import scipy.io as sio
from scipy import sparse
from scipy.optimize import minimize
import time
import math
import matplotlib as mpl
mpl.use('Agg')
import pylab
import matplotlib.pyplot as plt
from multiprocessing import Pool
from functools import partial
import random
import sys
import pickle

# Set of apertures starting with 16 that are well spread out.
kappa = [6, 17, 28, 39, 50, 61, 72, 83, 94, 105, 116, 127, 138, 149, 160, 171, 11, 22, 33, 44, 55, 66, 77, 88, 99, 110, 121, 132, 143, 154, 165, 1, 175, 14, 25, 36, 47, 58, 69, 80, 91, 102, 113, 124, 135, 146, 157, 168, 3, 8, 19, 30, 41, 52, 63, 74, 85, 96, 107, 118, 129, 140, 151, 162, 172, 176, 0, 2, 4, 5, 7, 9, 10, 12, 13, 15, 16, 18, 20, 21, 23, 24, 26, 27, 29, 31, 32, 34, 35, 37, 38, 40, 42, 43, 45, 46, 48, 49, 51, 53, 54, 56, 57, 59, 60, 62, 64, 65, 67, 68, 70, 71, 73, 75, 76, 78, 79, 81, 82, 84, 86, 87, 89, 90, 92, 93, 95, 97, 98, 100, 101, 103, 104, 106, 108, 109, 111, 112, 114, 115, 117, 119, 120, 122, 123, 125, 126, 128, 130, 131, 133, 134, 136, 137, 139, 141, 142, 144, 145, 147, 148, 150, 152, 153, 155, 156, 158, 159, 161, 163, 164, 166, 167, 169, 170, 173, 174, 177]

## Where the data is stored
IntheOffice = '/mnt/datadrive/Data/DataProject/HN/'
AtHome = '/home/wilmer/Dropbox/MATdata/DataProject/HN/'
readfolder = AtHome
## subfolder that contains the dose matrices
readfolderD = readfolder + 'Dij/'
## Folder where to output results
outputfolder = '/home/wilmer/Dropbox/Research/IMRTOptimizer/output/'
## This is where objectives are located for the objective function to be minimized
objfile = readfolder + 'obj1.txt'
## File describing what structures are targets and what structures are OAR's
structurefile = '/home/wilmer/Dropbox/IpOptSolver/TestData/HNdata/structureInputs.txt'
## Fila that contains basic algorithm inputs (not implemented)
algfile = '/home/wilmer/Dropbox/IpOptSolver/TestData/HNdata/algInputsWilmer.txt'
## Priority list of Organs of Interest. The 1 is subtracted at read time so the user doesn't have to do it everytime
priority = [7, 24, 25, 23, 22, 21, 20, 16, 15, 14, 13, 12, 10, 11, 9, 4, 3, 1, 2, 17, 18, 19, 5, 6, 8]
priority = (np.array(priority)-1).tolist()
eliminationThreshold = 10E-3
kappasize = 16
## This is the number of cores to use
numcores = 8

## Whether this run will include the whole circle
WholeCircle = False

## Initial Angle
gastart = 0 ;

## Final Angle
gaend = 356;
if WholeCircle:
    gastep = 2;
else:
    ## Change this value and set WholeCircle to false if you just want to debug a subset of the data
    gastep = 10;
## This vector allows users to convert locations into degree angles.
pointtoAngle = range(gastart, gaend, gastep)

ga=[];
ca=[];

## Contains all information relevant to a particular region, how many voxels and their location in relevant arrays.
class region:
    ## index used for identification
    index = int()
    ## Size in voxels of this particular structure
    sizeInVoxels = int()
    ## vector of maskvalue-indicated indices for non-overlapping structure assignment
    indices = np.empty(1, dtype=int)
    ## vector of fullmaskvalue-indicated indicies for complete structure assignment
    fullIndices = np.empty(1,dtype=int)
    ## True if target
    target = False
    ## Class initializer
    def __init__(self, iind, iindi, ifullindi, itarget):
        self.index = iind
        self.sizeInVoxels = len(iindi)
        self.indices = iindi
        self.fullIndices = ifullindi
        self.target = itarget

## apertureList is a class definition of locs and angles that is always sorted.
# Its attributes are loc which is the numeric location; It has range 0 to 178 for
# the HN case; Angle is the numeric angle in degrees; It ranges from 0 to 358 degrees
# apertureList should be sorted in ascending order everytime you add a new element; User CAN make this safe assumption
class apertureList:
    ## constructor initializes empty lists
    def __init__(self):
        ## Location in index range(0,numbeams)
        self.loc = []
        ## Angles ranges from 0 to 360
        self.angle = []
    ## Insert a new angle in the list of angles to analyse.
    # Gets angle information and inserts location and angle
    # In the end it sorts the list in increasing order
    def insertAngle(self, i, aperangle):
        self.angle.append(aperangle)
        self.loc.append(i)
        # Sort the angle list in ascending order
        self.loc.sort()
        self.angle.sort()
    ## Removes the index and its corresponding angle from the list.
    # Notice that it only removes the first occurrence; but if you have done everything correctly this should never
    # be a problem
    def removeIndex(self, index):
        toremove = [i for i,x in enumerate(self.loc) if x == index]
        self.loc.pop(toremove[0]) # Notice that it removes the first entry
        self.angle.pop(toremove[0])
    ## Looks for the angle and removes the index and the angle corresponding to it from the list
    def removeAngle(self, tangl):
        toremove = [i for i,x in enumerate(self.angle) if x == tangl]
        self.loc.pop(toremove[0])
        self.angle.pop(toremove[0])
    ## Overloads parenthesis operator in order to fetch the ANGLE given an index.
    # Returns the angle at the ith location given by the index.
    # First Find the location of that index in the series of loc
    # Notice that this function overloads the parenthesis operator for elements of this class.
    def __call__(self, index):
        toreturn = [i for i,x in enumerate(self.loc) if x == index]
        return(self.angle[toreturn[0]])
    ## Returns the length of this instantiation without the need to pass parameters.
    def len(self):
        return(len(self.loc))
    ## Returns True if the list is empty; otherwise returns False.
    def isEmpty(self):
        if 0 == len(self.loc):
            return(True)
        else:
            return(False)

## This defines the global VMAT class that contains most of the VMAT data to be used in the implementation
# Most of the values were defined as static attributes and only one instantiation at a time is possible. But this should not
# be a problem. The file also contains functions to be used when you call the optimizer.
class vmat_class:
    ## number of beamlets
    numX = 0
    ## number of voxels in the small voxel space
    numvoxels = int()
    ## number of structures/regions
    numstructs = 0
    ## number of organs at risk
    numoars = 0
    ## num of targets
    numtargets = 0
    ## num of beams
    numbeams = 0
    ## num of nonzeros in Dij matrix
    totaldijs = 0
    ## objectiveValue of the final function
    objectiveValue = float("inf")
    ## number of beamlets per beam
    beamletsPerBeam = []
    ## number of nonzeroes in Dij per beam
    dijsPerBeam = []
    ## non-overlapping mask value per voxel
    maskValue = []
    ## complete mask value per voxel ( A voxel may cover more than one structure = OAR's + T's)
    fullMaskValue = []
    ## index values of structures in region list (should be 0,1,etc)
    regionIndices = []
    ## region indices of target structures (from region vector)
    targets = []
    ## region indices of oars
    oars = []
    ## vector of regions (holds structure information)
    regions = []
    ## vector of data input files for objectives
    objectiveInputFiles = []
    ## vector of data input files for constraints
    constraintInputFiles = []
    ## vector of data input for algorithm options
    algOptions = []
    ## Holds function data parameters from objectFunctioninput file
    functionData = []
    voxelAssignment = []
    ## List of apertures not yet selected
    notinC = apertureList()
    ## List of apertures already selected
    caligraphicC = apertureList()

    ## varios folders
    outputDirectory = ""# given by the user in the first lines of *.py
    dataDirectory = ""

    # dose variables
    currentDose = np.empty(1) # dose variable
    currentIntensities = np.empty(1)

    ## this is the intersection of all beamlets geographical locations in centimeters
    ## It is unique for each value in the x coordinate axis. Beamlet data is organized first in the X axis and then
    # moves onto the Y axis
    xinter = []
    ## Same as xinter but for y axis
    yinter = []

    ## This is a list of lists; There is one for each aperture angle and it contains the x coordinate of each of the
    # nonzero available beamlets
    xdirection = []
    ## Same as xdirection but in the y coordinate
    ydirection = []

    ## List of lists. Contains limits on the left side of the aperture
    llist = []
    ## List of lists. Contains limits on the right side of the aperture
    rlist = []
    ## Results from latest restricted master problem
    rmpres = []

    ## Gradient in voxel dimensions This is the gradient of dF / dZ. Dimension is numvoxels
    voxelgradient = []

    ## This is the gradient of dF / dk. Dimension is num Apertures
    aperturegradient = []
    ## Contains the location of the beamlets that belong to open aperture and that have data available.
    openApertureMaps = []
    ## Contains data to create the diagonal matrices to process the gradient
    diagmakers = []
    ## Contains the strengths of the beamlet. Has same length as diagmakers and openApertureMaps
    strengths = []
    dZdK = 0.0
    ## This vector produces the angle corresponding to a particular index
    pointtoAngle = []
    ## This list contains the Dose to Point matrices for each of the beam angles.
    Dlist = []
    ## This function returns the objective value and the gradient

    ## Counts how many times I have entered the aperture replacement option
    entryCounter = 0
    ## Maintain a list of apertures removed each iteration using the removal criterion
    listIndexofAperturesRemovedEachStep = []
    ## This function regularly enters the optimization engine to calculate the dose; Used for objective function
    def calcDose(self):
        self.currentDose = np.zeros(self.numvoxels, dtype = float)
        # dZdK will have a dimension that is numvoxels x numbeams
        self.dZdK = np.matrix(np.zeros((self.numvoxels, self.numbeams)))
        if self.caligraphicC.len() != 0:
            for i in self.caligraphicC.loc:
                ## D[:, oam] * diag * R^{oam x numbeamlets}
                self.currentDose += DlistT[i][:,self.openApertureMaps[i]] * sparse.diags(self.strengths[i]) * np.repeat(self.currentIntensities[i], len(self.openApertureMaps[i]), axis = 0)
                self.dZdK[:,i] = (DlistT[i] * sparse.diags(self.diagmakers[i], 0)).sum(axis=1)

    ## This function regularly enters the optimization engine to calculate objective function and gradients
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

        # Notice that I use two types of gradients. One for voxels and one for apertures
        self.voxelgradient = 2 * (oDoseObjGl - uDoseObjGl)
        self.aperturegradient = (np.asmatrix(self.voxelgradient) * self.dZdK).transpose()

    # default constructor
    def __init__(self):
        self.numX = 0

########## END OF CLASS DECLARATION ###########################################

# First of all make sure that I can read the data

# In the data directory with the *VOILIST.mat files, this opens up
# each structure file and reads in the structure names and sizes in
# voxels

start = time.time()

## This class contains all variables. At this point it is implemented as a class with only static variables. This will
# change in figure versions.
data = vmat_class()

data.outputDirectory = outputfolder # given by the user in the first lines of *.pydoc
data.dataDirectory = readfolder
data.pointtoAngle = pointtoAngle
# Function definitions
####################################################################

## This function returns a dictionary with the dimension in voxel
# units for x,y,z axis. This function uses a unique txt file assigned by the
def readctvoxelinfo():
    lines = [line.rstrip('\n') for line in open(readfolder + 'CTVOXEL_INFO.txt', 'r')]
    tempocoor = []
    for i in range(0,3):
        tempocoor.append(int(lines[i].rsplit(None, 1)[-1]))
    coordims = dict(x=tempocoor[0],y=tempocoor[1],z=tempocoor[2])
    return(coordims)

oldfolder = os.getcwd()
os.chdir(readfolder)
## Keep a log of all volumes of interest
allFiles = glob.glob("*VOILIST.mat")
allBeamInfos = glob.glob("*Couch0_BEAMINFO.mat")
## Will contain all VOI names
allNames = sorted(allFiles) #Make sure it's sorted because it was not.
allBeamInfoNames = sorted(allBeamInfos)
## How many structures of interest are included in this case.
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
    bigZ[Vorg[s]] = 1.0 # Here I'm assigning a 1 to those voxels that belong in small voxel space also.

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
        ## If big space voxel is nonzero, save to small vxl space
        data.voxelAssignment[counter] = i
        counter+=1
print('mapping from small voxel space to big voxel space done')

# originalVoxels is the mapping from big voxel space to small voxel
# space. It is VERY important to initialize originalVoxels with NAN in this case.
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

# Below, stands which organ the voxel belongs to.
## Full masking value using 64 bits (Up to 48 structures)
maskValueFull = np.zeros(nVox.astype(np.int64))
## Most important masking value only.
maskValueSingle = np.zeros(nVox.astype(np.int64))
# this priority is the order of priority for assigning a single structure per
# voxel (from least to most important)

for i in range(0, numStructs):
    s = priority[i]
    # generates mask values (the integer that we decompose to get structure
    # assignment). for single it just overrides with the more important
    # structure
    maskValueFull[originalVoxels[Vorg[s]].astype(int)] = maskValueFull[originalVoxels[Vorg[s]].astype(int)]+2**(s)
    maskValueSingle[originalVoxels[Vorg[s]].astype(int)] = 2**(s)

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
nDIJSPB = np.zeros(len(ga))

###############################################################################
## Beginning of Troy's cpp code (interpreted, not copied)
## A comment
## This comes from first two lines in doseInputs txt file (troy's version)
## CAREFUL. Now numvoxels refers to small voxels space
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
    data.openApertureMaps.append([]) #Just start an empty map of apertures
    data.diagmakers.append([])
    data.strengths.append([])
## After reading the beaminfo information. Read CUT the data.

## Number of beamlets in a row
N = len(data.yinter) #N will be related to the Y axis.
## Number of rows in an aperture
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
data.Dlist = [None] * data.numbeams
DlistT = [None] * data.numbeams

## This function reads the dose to points matrices. It also cuts them since extended versions of the algorithm cannot
# be implemented yet.
def readDmatrix(i):
    fname = 'Gantry' + str(ga[i]) + '_Couch' + str(0) + '_D.mat'
    print('Reading matrix from gantry & couch angle: ' + fname)
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
    return(i, Dlittle)

# Read the data in parallel
if __name__ == '__main__':
    pool = Pool(processes=6)              # process per MP
    Allmats = pool.map(readDmatrix, range(0, data.numbeams))

# Assign data
for objResult in Allmats:
    data.Dlist[objResult[0]] = objResult[1]

print('Finished reading D matrices')

for i in range(0, data.numbeams):
    # ininter will contain the beamlet directions that belong in the intersection of all apertures
    ininter = []
    for j in range(0, len(data.xdirection[i])):
        if (data.xdirection[i][j] in data.xinter and data.ydirection[i][j] in data.yinter):
            ininter.append(j)

    # Once I have ininter I will cut all the elements that are
    data.xdirection[i] = data.xdirection[i][ininter]
    data.ydirection[i] = data.ydirection[i][ininter]
    print(type(data.Dlist[i]))
    data.Dlist[i] = data.Dlist[i][ininter,:]
    DlistT[i] = data.Dlist[i].transpose()
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

def calcObjGrad(x, user_data = None):
    data.currentIntensities = x
    data.calcDose()
    data.calcGradientandObjValue()
    return(data.objectiveValue, data.aperturegradient)

## Find geographical location of the ith row in aperture index given by index.
# Input:    i:     Row
#           index: Index of this aperture
# Output:   validbeamlets ONLY contains those beamlet INDICES for which we have available data in this beam angle
#           validbeamletspecialrange is the same as validbeamlets but appending the endpoints
def fvalidbeamlets(i, index):
    geolocX = data.xinter[i] # geolocx in centimeteres. This is coordinate x of beamlet location.
    # Find all possible locations of beamlets in this row according to geographical location
    indys = np.where(geolocX == data.xdirection[index])
    # In centimeters. This is coordinate y of all beamlets with x coordinate == geolocX
    # Notice that geolocYs ONLY contains those beamlets that are available. As opposed to yinter which contains all.
    geolocYs = data.ydirection[index][indys]

    validbeamletlogic = np.in1d(data.yinter, geolocYs)
    # validbeamlets ONLY contains those beamlets for which we have available data in this beam angle in index coordinates
    validbeamlets = np.array(range(0, len(data.yinter)))[validbeamletlogic]
    validbeamletspecialrange = np.append(np.append(min(validbeamlets) - 1, validbeamlets), max(validbeamlets) + 1)
    # That last line appends the endpoints.
    return(validbeamlets, validbeamletspecialrange)

## C, C2, C3 are constants in the penalization function
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
# thisApertureIndex = index location in the set of apertures that I have saved.
def PPsubroutine(C, C2, C3, b, angdistancem, angdistancep, vmax, speedlim, predec, succ, N, M, thisApertureIndex, bw):
    D = data.Dlist[thisApertureIndex]
    # vmaxm and vmaxp describe the speeds that are possible for the leaves from the predecessor and to the successor
    vmaxm = vmax
    vmaxp = vmax
    # Arranging the predecessors and the succesors.
    #Predecessor left and right indices
    if type(predec) is list:
        lcm = [-1] * M
        rcm = [N] * M
        # If there is no predecessor is as if the pred. speed was infinite
        vmaxm = float("inf")
    else:
        lcm = data.llist[predec]
        rcm = data.rlist[predec]

    #Succesors left and right indices
    if type(succ) is list:
        lcp = [-1] * M
        rcp = [N] * M
        # If there is no successor is as if the succ. speed was infinite.
        vmaxp = float("inf")
    else:
        lcp = data.llist[succ]
        rcp = data.rlist[succ]

    validbeamlets, validbeamletspecialrange = fvalidbeamlets(0, thisApertureIndex)
    # First handle the calculations for the first row
    beamGrad = D * data.voxelgradient
    # Keep the location of the most leaf

    nodesinpreviouslevel = 0
    oldflag = 0
    posBeginningOfRow = 1
    thisnode = 0
    # Max beamlets per row
    bpr = 50
    networkNodesNumber = bpr * bpr + M * bpr * bpr + bpr * bpr # An overestimate of the network nodes in this network
    # Initialization of network vectors. This used to be a list before
    lnetwork = np.zeros(networkNodesNumber, dtype = np.int) #left limit vector
    rnetwork = np.zeros(networkNodesNumber, dtype = np.int) #right limit vector
    mnetwork = np.ones(networkNodesNumber, dtype = np.int) #Only to save some time in the first loop
    wnetwork = np.empty(networkNodesNumber, dtype = np.float) # Weight Vector
    wnetwork[:] = np.inf
    dadnetwork = np.zeros(networkNodesNumber, dtype = np.int) # Dad Vector. Where Dad is the combination of (l,r) in previous row
    # Work on the first row perimeter and area values
    leftrange = range(math.ceil(max(-1, lcm[0] - vmaxm * (angdistancem/speedlim)/bw , lcp[0] - vmaxp * (angdistancep/speedlim)/bw )), 1 + math.floor(min(N - 1, lcm[0] + vmaxm * (angdistancem/speedlim)/bw , lcp[0] + vmaxp * (angdistancep/speedlim)/bw )))
    # Check if unfeasible. If it is then assign one value but tell the result to the person running this
    if (161 == thisApertureIndex):
        print(' aperture ' + str(thisApertureIndex), 'ERROR Report: angdistancem, angdistancep', angdistancem, angdistancep, '\nFull left limits:', 'predecesor: ', predec, 'succesor: ', succ)
        print('right limits before:', rcm)
        print('right limits after: ', rcp)
        print('left limits before: ', lcm)
        print('left limits after   ', lcp)
    if (0 == len(leftrange)):
        midpoint = (angdistancep * lcm[0] + angdistancem * lcp[0])/(angdistancep + angdistancem)
        leftrange = np.arange(midpoint, midpoint + 1)
        ##print('constraint leftrange at level ' + str(0) + ' aperture ' + str(thisApertureIndex) + ' could not be met', 'ERROR Report: lcm[0], angdistancem, lcp[0], angdistancep', lcm[0], angdistancem, lcp[0], angdistancep, '\nFull left limits, lcp, lcm:', lcp, lcm, 'm: ', 0, 'predecesor: ', predec, 'succesor: ', succ)
    for l in leftrange:
        rightrange = range(math.ceil(max(l + 1, rcm[0] - vmaxm * (angdistancem/speedlim)/bw , rcp[0] - vmaxp * (angdistancep/speedlim)/bw )), 1 + math.floor(min(N, rcm[0] + vmaxm * (angdistancem/speedlim)/bw , rcp[0] + vmaxp * (angdistancep/speedlim)/bw )))
        if (0 == len(rightrange)):
            midpoint = (angdistancep * rcm[0] + angdistancem * rcp[0])/(angdistancep + angdistancem)
            rightrange = np.arange(midpoint, midpoint + 1)
            ##print('constraint rightrange at level ' + str(0) + ' aperture ' + str(thisApertureIndex) + ' could not be met', 'ERROR Report: lcm[0], angdistancem, lcp[0], angdistancep', lcm[0], angdistancem, lcp[0], angdistancep, '\nFull left limits, rcp, rcm:', rcp, rcm, 'm: ', 0, 'predecesor: ', predec, 'succesor: ', succ)
        for r in rightrange:
            thisnode = thisnode + 1
            nodesinpreviouslevel = nodesinpreviouslevel + 1
            # First I have to make sure to add the beamlets that I am interested in
            if(l + 1 < r): # prints r numbers starting from l + 1. So range(3,4) = 3
                ## Take integral pieces of the dose component
                possiblebeamletsthisrow = np.intersect1d(range(int(np.ceil(l+1)),int(np.floor(r))), validbeamlets) - min(validbeamlets)
                ## Calculate dose on the sides
                DoseSide = -((np.ceil(l+1) - (l+1)) * beamGrad[int(np.floor(l+1))] + (r - np.floor(r)) * beamGrad[int(np.ceil(r))])
                if (len(possiblebeamletsthisrow) > 0):
                    Dose = -beamGrad[ possiblebeamletsthisrow ].sum()
                    weight = C * ( C2 * (r - l) - C3 * b * (r - l)) - Dose + 10E-10 * (r-l) + DoseSide# The last term in order to prefer apertures opening in the center
                else:
                    weight = C * ( C2 * (r - l) - C3 * b * (r - l)) + 10E-10 * (r-l) + DoseSide
            else:
                weight = 0.0
            # Create node (1,l,r) in array of existing nodes and update the counter
            # Replace the following expression
            lnetwork[thisnode] = l
            rnetwork[thisnode] = r
            wnetwork[thisnode] = weight
            # dadnetwork and mnetwork don't need to be changed here for obvious reasons
    posBeginningOfRow = posBeginningOfRow + nodesinpreviouslevel
    leftmostleaf = len(validbeamlets) - 1 # Position in python position(-1) of the leftmost leaf
    # Then handle the calculations for the m rows. Nodes that are neither source nor sink.
    for m in range(1,M):
        # Get the beamlets that are valid in this row in particular (all others are still valid but are zero)
        validbeamlets, validbeamletspecialrange = fvalidbeamlets(m, thisApertureIndex)
        oldflag = nodesinpreviouslevel
        nodesinpreviouslevel = 0
        # And now process normally checking against valid beamlets
        leftrange = range(math.ceil(max(-1, lcm[m] - vmaxm * (angdistancem/speedlim)/bw , lcp[m] - vmaxp * (angdistancep/speedlim)/bw )), 1 + math.floor(min(N - 1, lcm[m] + vmaxm * (angdistancem/speedlim)/bw , lcp[m] + vmaxp * (angdistancep/speedlim)/bw )))
        # Check if unfeasible. If it is then assign one value but tell the result to the person running this
        if(0 == len(leftrange)):
            midpoint = (angdistancep * lcm[m] + angdistancem * lcp[m])/(angdistancep + angdistancem)
            leftrange = np.arange(midpoint, midpoint + 1)
            ##print('constraint leftrange at level ' + str(m) + ' aperture ' + str(thisApertureIndex) + ' could not be met', 'ERROR Report: lcm[m], angdistancem, lcp[m], angdistancep', lcm[m], angdistancem, lcp[m], angdistancep, '\nFull left limits, lcp, lcm:', lcp, lcm, 'm: ', m, 'predecesor: ', predec, 'succesor: ', succ)
            ##leftrange = range(math.ceil(max(-1, lcm[m] - vmaxm * (angdistancem/speedlim)/bw , lcp[m] - vmaxp * (angdistancep/speedlim)/bw )), 1 + math.ceil(max(-1, lcm[m] - vmaxm * (angdistancem/speedlim)/bw , lcp[m] - vmaxp * (angdistancep/speedlim)/bw )))
        for l in leftrange:
            rightrange = range(math.ceil(max(l + 1, rcm[m] - vmaxm * (angdistancem/speedlim)/bw , rcp[m] - vmaxp * (angdistancep/speedlim)/bw )), 1 + math.floor(min(N, rcm[m] + vmaxm * (angdistancem/speedlim)/bw , rcp[m] + vmaxp * (angdistancep/speedlim)/bw )))
            if (0 == len(rightrange)):
                midpoint = (angdistancep * rcm[m] + angdistancem * rcp[m])/(angdistancep + angdistancem)
                rightrange = np.arange(midpoint, midpoint + 1)
                ##print('constraint rightrange at level ' + str(m) + ' aperture ' + str(thisApertureIndex) + ' could not be met', 'ERROR Report: lcm[m], angdistancem, lcp[m], angdistancep', lcm[m], angdistancem, lcp[m], angdistancep, '\nFull left limits, rcp, rcm:', rcp, rcm, 'm: ', m, 'predecesor: ', predec, 'succesor: ', succ)
            for r in rightrange:
                nodesinpreviouslevel = nodesinpreviouslevel + 1
                thisnode = thisnode + 1
                # Create node (m, l, r) and update the level counter
                lnetwork[thisnode] = l
                rnetwork[thisnode] = r
                mnetwork[thisnode] = m
                wnetwork[thisnode] = np.inf
                # Select only those beamlets that are possible in between the (l,r) limits.
                possiblebeamletsthisrow = np.intersect1d(range(int(np.ceil(l+1)),int(np.floor(r))), validbeamlets) + leftmostleaf - min(validbeamlets)
                DoseSide = -((np.ceil(l+1) - (l+1)) * beamGrad[int(np.floor(l+1))] + (r - np.floor(r)) * beamGrad[int(np.ceil(r))])
                if(len(possiblebeamletsthisrow) > 0):
                    Dose = -beamGrad[possiblebeamletsthisrow].sum()
                    C3simplifier = C3 * b * (r - l)
                else:
                    Dose = 0.0
                    C3simplifier = 0.0
                lambdaletter = np.absolute(lnetwork[(posBeginningOfRow - oldflag): posBeginningOfRow] - l) + np.absolute(rnetwork[(posBeginningOfRow - oldflag): posBeginningOfRow] - r) - 2 * np.maximum(0, lnetwork[(posBeginningOfRow - oldflag): posBeginningOfRow] - r) - 2 * np.maximum(0, l - np.absolute(rnetwork[(posBeginningOfRow - oldflag): posBeginningOfRow]))
                weight = C * (C2 * lambdaletter - C3simplifier) - Dose  + 10E-10 * (r-l) + DoseSide # The last term in order to prefer apertures opening in the center
                # Add the weights that were just calculated
                newweights = wnetwork[(posBeginningOfRow - oldflag): posBeginningOfRow] + weight
                # Find the minimum and its position in the vector
                minloc = np.argmin(newweights)
                wnetwork[thisnode] = newweights[minloc]
                dadnetwork[thisnode] = minloc + posBeginningOfRow - oldflag

        posBeginningOfRow = nodesinpreviouslevel + posBeginningOfRow # This is the total number of network nodes
        # Keep the location of the leftmost leaf
        if (161 == thisApertureIndex):
            print('posbeginningofrow, nodesinpreviouslevel and m', posBeginningOfRow, nodesinpreviouslevel, m)
        leftmostleaf = len(validbeamlets) + leftmostleaf
    # thisnode gets augmented only 1 because only the sink node will be added
    thisnode = thisnode + 1
    if (161 == thisApertureIndex):
        print('wnetwork[thisnode]', wnetwork[thisnode])
        print('posbeginingofrow', posBeginningOfRow, 'nodesinpreviouslevel', nodesinpreviouslevel)
        print(range(posBeginningOfRow - nodesinpreviouslevel, posBeginningOfRow))
    for mynode in (range(posBeginningOfRow - nodesinpreviouslevel, posBeginningOfRow )): # +1 because otherwise it could be empty
        weight = C * ( C2 * (rnetwork[mynode] - lnetwork[mynode] ))
        if (161 == thisApertureIndex):
            print('wnetwork[mynode] and weight:', wnetwork[mynode], weight)
        if(wnetwork[mynode] + weight <= wnetwork[thisnode]):
            wnetwork[thisnode] = wnetwork[mynode] + weight
            dadnetwork[thisnode] = mynode
            p = wnetwork[thisnode]
    thenode = thisnode # WILMER take a look at this
    l = []
    r = []
    while(1):
        if (161 == thisApertureIndex):
            print('here is node:', thenode)
        # Find the predecessor data
        l.append(lnetwork[thenode])
        r.append(rnetwork[thenode])
        thenode = dadnetwork[thenode]
        if(0 == thenode): # If at the origin then break
            break
    l.reverse()
    r.reverse()
    #Pop the last elements because this is the direction of nonexistent sink field
    l.pop(); r.pop()
    if(thisApertureIndex == 161):
        print('p al final: ', p)
    return(p, l, r)

def parallelizationPricingProblem(i, C, C2, C3, b, vmax, speedlim, N, M, bw):
    thisApertureIndex = i

    print("analysing available aperture" , thisApertureIndex)
    # Find the successor and predecessor of this particular element
    try:
        #This could be done with angles instead of indices (reconsider this at some point)
        succs = [i for i in data.caligraphicC.loc if i > thisApertureIndex]
    except:
        succs = []
    try:
        predecs = [i for i in data.caligraphicC.loc if i < thisApertureIndex]
    except:
        predecs = []

    # If there are no predecessors or succesors just return an empty list. If there ARE, then return the indices
    if 0 == len(succs):
        succ = []
        angdistancep = np.inf
    else:
        succ = min(succs)
        angdistancep = data.caligraphicC(succ) - data.notinC(thisApertureIndex)
    if 0 == len(predecs):
        predec = []
        angdistancem = np.inf
    else:
        predec = max(predecs)
        angdistancem = data.notinC(thisApertureIndex) - data.caligraphicC(predec)

    # Find Numeric value of previous and next angle.
    p, l, r = PPsubroutine(C, C2, C3, b, angdistancem, angdistancep, vmax, speedlim, predec, succ, N, M, thisApertureIndex, bw)
    return(p,l,r,thisApertureIndex)

def PricingProblem(C, C2, C3, b, vmax, speedlim, N, M, bw):
    pstar = np.inf
    bestAperture = None
    print("Choosing one aperture amongst the ones that are available")
    # Allocate empty list with enough size for all l, r combinations
    global lall
    global rall
    global pall
    lall = [None] * data.notinC.len()
    rall = [None] * data.notinC.len()
    pall = np.array([None] * data.notinC.len())

    partialparsubpp = partial(parallelizationPricingProblem, C=C, C2=C2, C3=C3, b=b, vmax=vmax, speedlim=speedlim, N=N, M=M, bw=bw)
    if __name__ == '__main__':
        pool = Pool(processes=numcores)              # process per MP
        respool = pool.map(partialparsubpp, data.notinC.loc)
    pool.close()
    pool.join()

    pvalues = np.array([result[0] for result in respool])
    indstar = np.argmin(pvalues)
    bestgroup = respool[indstar]
    pstar = bestgroup[0]
    lallret = bestgroup[1]
    rallret = bestgroup[2]
    bestApertureIndex = bestgroup[3]
    for i in range(0, len(respool)):
        if(M != len(respool[i][1])):
            sys.exit("Length of left vector is less than 28")
            print('aperture problem is:', i)

    print("Best aperture was: ", bestApertureIndex)
    return(pstar, lallret, rallret, bestApertureIndex)

def solveRMC(YU):
    ## IPOPT SOLUTION
    start = time.time()
    numbe = data.caligraphicC.len()
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
    # Create the boundaries making sure that the only free variables are the ones with perfectly defined apertures.
    boundschoice = []
    for thisindex in range(0, data.numbeams):
        if thisindex in data.caligraphicC.loc: #Only activate what is an aperture
            boundschoice.append((0, YU))
        else:
            boundschoice.append((0, 0))
    print(len(data.currentIntensities), len(boundschoice))
    res = minimize(calcObjGrad, data.currentIntensities, method='L-BFGS-B', jac = True, bounds = boundschoice, options={'ftol':1e-1, 'disp':5,'maxiter':200})

    print('Restricted Master Problem solved in ' + str(time.time() - start) + ' seconds')
    return(res)

# The next function prints DVH values
def printresults(iterationNumber, myfolder):
    numzvectors = 1
    maskValueFull = np.array([int(i) for i in data.fullMaskValue])
    print('Starting to Print Results')
    for i in range(0, numzvectors):
        zvalues = data.currentDose
        maxDose = max([float(i) for i in zvalues])
        dose_resln = 0.1
        dose_ub = maxDose + 10
        bin_center = np.arange(0,dose_ub,dose_resln)
        # Generate holder matrix
        dvh_matrix = np.zeros((data.numstructs, len(bin_center)))
        # iterate through each structure
        for s in range(0,data.numstructs):
            allNames[s] = allNames[s].replace("_VOILIST.mat", "")
            doseHolder = sorted(zvalues[[i for i,v in enumerate(maskValueFull & 2**s) if v > 0]])
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
    plt.title('Iteration: ' + str(iterationNumber))
    plt.legend(allNames)
    plt.savefig(myfolder + 'DVH-for-debugging-greedyVMAT.png')
    plt.close()

    voitoplot = [0, 18, 23, 17, 2, 8]
    dvhsub2 = dvh_matrix[voitoplot,]
    myfig2 = pylab.plot(bin_center, dvhsub2.T, linewidth = 1.0, linestyle = '--')
    plt.grid(True)
    plt.xlabel('Dose Gray')
    plt.ylabel('Fractional Volume')
    plt.title('VMAT Iteration:' + str(iterationNumber))
    #allNames.reverse()
    plt.legend([allNames[i] for i in voitoplot])
    plt.savefig(myfolder + 'DVH-at-Iteration-Subplot-for-debugging-greedyVMAT.png')
    plt.close()

def colGen(C, WholeCircle, initialApertures):
    C2 = 1.0
    C3 = 1.0
    eliminationPhase = False

    ## Maximum leaf speed
    vmax = 2.25      # 2.25 cms per second
    speedlim = 0.83  # Values are in the VMATc paper page 2968. 0.85 < s < 6
    beamletwidth = 0.5 # Width of the beamlet according to CORT pdf "Summary of Patient Characteristics"
    ## Maximum Dose Rate
    RU = 10.0
    ## Maximum intensity
    YU = RU / speedlim

    # Assign the most open apertures as initial apertures. They will not have any intensity applied to them.
    for i in range(0, data.numbeams):
        data.llist.append([-1] * len(data.xinter))
        data.rlist.append([len(data.yinter) + 1] * len(data.xinter))

    #Step 0 on Fei's paper. Set C = empty and zbar = 0. The gradient of numbeams dimensions generated here will not
    # be used, and therefore is nothing to worry about.
    # At the beginning no apertures are selected, and those who are not selected are all in notinC
    if WholeCircle:
        random.seed(13)
        global kappa
        kappa = random.sample(kappa, len(kappa))
        for j in range(0, min(initialApertures, len(kappa))):
            i = kappa[0]
            data.notinC.insertAngle(i, data.pointtoAngle[i])
            kappa.pop(0)
        print('apertures initial', data.notinC)
    else:
        # Initialize kappa with the values that are given to me only
        kappa = []
        i = 0
        for j in range(gastart, gaend, gastep):
            kappa.append(i)
            i = i + 1
        random.seed(13)
        kappa = random.sample(kappa, len(kappa))
        for j in range(0, min(initialApertures, len(kappa))):
            i = kappa[0]
            data.notinC.insertAngle(i, data.pointtoAngle[i])
            kappa.pop(0)
    data.caligraphicC = apertureList()
    pstar = -np.inf
    plotcounter = 0
    optimalvalues = []
    while (pstar < 0) & (data.notinC.len() > 0):
        # Step 1 on Fei's paper. Use the information on the current treatment plan to formulate and solve an instance of the PP
        data.calcDose()
        data.calcGradientandObjValue()
        pstar, lm, rm, bestApertureIndex = PricingProblem(C, C2, C3, 0.5, vmax, speedlim, N, M, beamletwidth)
        # Step 2. If the optimal value of the PP is nonnegative**, go to step 5. Otherwise, denote the optimal solution to the
        # PP by c and Ac and replace caligraphic C and A = Abar, k \in caligraphicC
        if pstar >= 0:
            #This choice includes the case when no aperture was selected
            print('Program finishes because no aperture was selected to enter')
            break
        else:
            # Update caligraphic C.
            data.caligraphicC.insertAngle(bestApertureIndex, data.notinC(bestApertureIndex))
            data.notinC.removeIndex(bestApertureIndex)
            # Solve the instance of the RMP associated with caligraphicC and Ak = A_k^bar, k \in
            data.llist[bestApertureIndex] = lm
            data.rlist[bestApertureIndex] = rm
            # Precalculate the aperture map to save times.
            data.openApertureMaps[bestApertureIndex], data.diagmakers[bestApertureIndex], data.strengths[bestApertureIndex] = updateOpenAperture(bestApertureIndex)
            rmpres = solveRMC(YU) # Solve Restricted Master Problem.
            data.rmpres = rmpres
            ## List of apertures that was removed in this iteration
            IndApRemovedThisStep = []
            for thisindex in range(0, data.numbeams):
                if thisindex in data.caligraphicC.loc: #Only activate what is an aperture
                    if (rmpres.x[thisindex] < eliminationThreshold) & (eliminationPhase):
                        ## Maintain a tally of apertures that are being removed
                        data.entryCounter += 1
                        IndApRemovedThisStep.append(thisindex)
                        # Remove from caligraphicC and add to notinC
                        data.notinC.insertAngle(thisindex, data.pointtoAngle[thisindex])
                        data.caligraphicC.removeIndex(thisindex)
            print('Indapremoved this step:')
            print(IndApRemovedThisStep)
            if len(data.listIndexofAperturesRemovedEachStep) > 1:
                ## Check if any element that I'm removing here was removed in the previous iteration and exit
                print('thisstep removed: ', IndApRemovedThisStep)
                print('removed previously: ', data.listIndexofAperturesRemovedEachStep)
                # Discuss this condition with Marina.
                if(np.any(np.in1d(IndApRemovedThisStep, data.listIndexofAperturesRemovedEachStep[len(data.listIndexofAperturesRemovedEachStep) - 1]))):
                    print('Program finishes because it keeps selecting the same aperture to add and delete')
                    break
                # Do the same with two steps back.
                if(np.any(np.in1d(IndApRemovedThisStep, data.listIndexofAperturesRemovedEachStep[len(data.listIndexofAperturesRemovedEachStep) - 2]))):
                    print('Program finishes because it keeps selecting the same aperture to add and delete')
                    break
            ## Save all apertures that were removed in this step
            data.listIndexofAperturesRemovedEachStep.append(IndApRemovedThisStep)
            optimalvalues.append(rmpres.fun)
            plotcounter = plotcounter + 1
            # Add everything from notinC to kappa
            while(False == data.notinC.isEmpty()):
                kappa.append(data.notinC.loc[0])
                data.notinC.removeIndex(data.notinC.loc[0])

            # Choose a random set of elements in kappa
            random.seed(13)
            elemstoinclude = random.sample(kappa, min(initialApertures, len(kappa)))
            for i in elemstoinclude:
                data.notinC.insertAngle(i, data.pointtoAngle[i])
                kappa.remove(i)
            # plotAperture(lm, rm, M, N, '/home/wilmer/Dropbox/Research/VMAT/VMATwPenCode/outputGraphics/', plotcounter, bestApertureIndex)
            printresults(plotcounter, '/home/wilmer/Dropbox/Research/VMAT/VMATwPenCode/outputGraphics0/')
            #Step 5 on Fei's paper. If necessary complete the treatment plan by identifying feasible apertures at control points c
            #notinC and denote the final set of fluence rates by yk
    plotApertures(C)
    ## Save to a pickle file and later plotting
    datasave = [data.numbeams, data.rmpres.x, C, C2, C3, vmax, speedlim, RU, YU, M, N, data.llist, data.rlist,
                data.fullMaskValue, data.currentDose, data.currentIntensities, data.numstructs, allNames,
                data.objectiveValue, quadHelperThresh, quadHelperOver, quadHelperUnder, data.regionIndices,
                data.targets, data.oars]
    PIK = "/home/wilmer/Dropbox/Research/VMAT/VMATwPenCode/outputGraphics/pickle-C-" + str(C) + "-WholeCirCle-" + str(WholeCircle) + "-Kappa-" + str(kappasize) + "-save.dat"
    with open(PIK, "wb") as f:
        pickle.dump(datasave, f, pickle.HIGHEST_PROTOCOL)
    f.close()
    return(pstar)

## This function returns the set of available AND open beamlets for the selected aperture (i).
# The idea is to have everything ready and pre-calculated for the evaluation of the objective function in
# calcDose
# input: i is the index number of the aperture that I'm working on
# output: openaperturenp. the set of available AND open beamlets for the selected aperture. Doesn't contain fractional values
#         diagmaker. A vector that has a 1 in each position where an openaperturebeamlet is available.
# openaperturenp is read as openapertureMaps. A member of the VMAT_CLASS.
def updateOpenAperture(i):
    leftlimits = 0
    openaperture = []
    ## While openaperturenp contains positions, openapertureStrength contains proportion of the beamlets that's open.
    openapertureStrength = []
    diagmaker = np.zeros(data.Dlist[i].shape[0], dtype = float)
    for m in range(0, len(data.llist[i])):
        # Find geographical values of llist and rlist.
        # Find geographical location of the first row.
        validbeamlets, validbeamletspecialrange = fvalidbeamlets(m, i)
        # First index in this row (only full beamlets included in this part

        ## Notice that indleft and indright below may be floats instead of just integers
        if (data.llist[i][m] >= min(validbeamlets) - 1):
            ## I subtract min validbeamlets bec. I want to find coordinates in available space
            ## indleft is where the edge of the left leaf ends. From there on there are photons.
            indleft = data.llist[i][m] + 1 + leftlimits - min(validbeamlets)
        else:
            # if the left limit is too far away to the left, just take what's available
            indleft = 0

        if (data.rlist[i][m] > max(validbeamlets)):
            # If the right limit is too far to the left, just grab the whole thing.
            indright = len(validbeamlets) + leftlimits
        else:
            if(data.rlist[i][m] >= min(validbeamlets)):
                ## indright is where the edgo of the right leaf ends. From there on there are photons
                indright = data.rlist[i][m] - 1 + leftlimits - min(validbeamlets)
            else:
                # Right limit is to the left of validbeamlets (This situation is weird)
                indright = 0

        # Keep the location of the letftmost leaf
        leftlimits = leftlimits + len(validbeamlets)
        #print('indleft, data.llist[i][m], leftlimits, validbeam', indleft, data.llist[i][m], leftlimits, validbeamlets)
        if (np.floor(indleft) < np.ceil(indright)): ## Just a necessary logical check.
            first = True
            for thisbeamlet in range(int(np.floor(indleft)), int(np.ceil(indright))):
                strength = 1.0
                if first:
                    first = False
                    # Fix the proportion of the left beamlet that is open
                    strength = np.ceil(indleft) - indleft
                openapertureStrength.append(strength)
                diagmaker[thisbeamlet] = strength
                openaperture.append(thisbeamlet)
            ## Fix the proportion of the right beamlet that is open.
            strength = indright - np.floor(indright)
            if strength > 0.01:
                ## Important: There is no need to check if there exists a last element because after all, you already
                # checked whe you entered the if loop above this one
                openapertureStrength[-1] = strength
                diagmaker[-1] = strength

            ## One last scenario. If only a little bit of the aperture is open (less than a beamlet and within one beamlet
            if 1 == int(np.ceil(indright)) - int(np.floor(indleft)):
                strength = indright - indleft
                openapertureStrength[-1] = strength
                diagmaker[-1] = strength
    openaperturenp = np.array(openaperture, dtype=int) #Contains indices of open beamlets in the aperture
    return(openaperturenp, diagmaker, openapertureStrength)

def plotApertures(C):
    magnifier = 100
    ## Plotting apertures
    xcoor = math.ceil(math.sqrt(data.numbeams))
    ycoor = math.ceil(math.sqrt(data.numbeams))
    nrows, ncols = M,N
    print('numbeams', data.numbeams)
    YU = (10.0 / 0.83)
    for mynumbeam in range(0, data.numbeams):
        lmag = data.llist[mynumbeam]
        rmag = data.rlist[mynumbeam]
        ## Convert the limits to hundreds.
        for posn in range(0, len(lmag)):
            lmag[posn] = int(magnifier * lmag[posn])
            rmag[posn] = int(magnifier * rmag[posn])
        image = -1 * np.ones(magnifier * nrows * ncols)
            # Reshape things into a 9x9 grid
        image = image.reshape((nrows, magnifier * ncols))
        for i in range(0, M):
            image[i, lmag[i]:(rmag[i]-1)] = data.rmpres.x[mynumbeam]
        image = np.repeat(image, magnifier, axis = 0) # Repeat. Otherwise the figure will look flat like a pancake
        image[0,0] = YU # In order to get the right list of colors
        # Set up a location where to save the figure
        fig = plt.figure(1)
        plt.subplot(ycoor,xcoor, mynumbeam + 1)
        cmapper = plt.get_cmap("autumn_r")
        cmapper.set_under('black', 1.0)
        plt.imshow(image, cmap = cmapper, vmin = 0.0, vmax = YU)
        plt.axis('off')
    fig.savefig('/home/wilmer/Dropbox/Research/VMAT/VMATwPenCode/outputGraphics/plotofapertures'+ str(C) + '.png')

print('Preparation time took: ' + str(time.time() - start) + ' seconds')

before = time.time()
# Do the dirty work of calling the function and collect the results to be saved
for iter in np.arange(0, 1, 1):
    pstar = colGen(iter, WholeCircle, kappasize)
    ## Reinitialize everything for the next iteration
    data.objectiveValue = np.inf
    data.currentDose = np.empty(1) # dose variable
    data.currentIntensities = np.zeros(data.numbeams, dtype = float)
    data.llist = []
    data.rlist = []
    data.rmpres = []
    data.voxelgradient = []
    data.aperturegradient = []
    ## List of apertures not yet selected
    data.notinC = apertureList()
    ## List of apertures already selected
    data.caligraphicC = apertureList()
    data.openApertureMaps = []
    data.diagmakers = []
    data.strengths = []
    for i in range(0, data.numbeams):
        data.openApertureMaps.append([])
        data.diagmakers.append([])
        data.strengths.append([])
    data.dZdK = 0.0
    data.entryCounter = 0
    data.listIndexofAperturesRemovedEachStep = []
after = time.time()

print("The whole process took: " , after - before)
print('The whole program took: '  + str(time.time() - start) + ' seconds to finish')
print('You removed apertures using the removal criterion a total of: ', data.entryCounter, ' times')
print("You have graciously finished running this program")
