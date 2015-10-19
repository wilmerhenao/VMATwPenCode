
#!/usr/bin/python

import glob, os
import pyipopt
import numpy as np
import scipy.io as sio
from scipy import sparse

# First of all make sure that I can read the data

# In the data directory with the *VOILIST.mat files, this opens up
# each structure file and reads in the structure names and sizes in
# voxels

readfolder = '/home/wilmer/Documents/Troy_BU/Data/DataProject/HN/'
readfolderD = readfolder + 'Dij/'
outputfolder = '/home/wilmer/Dropbox/Research/VMAT/output/'
degreesep = 60 # How many degrees in between separating neighbor beams.
objfile = '/home/wilmer/Dropbox/IpOptSolver/TestData/HNdata/objectives/obj1.txt'
structurefile = '/home/wilmer/Dropbox/IpOptSolver/TestData/HNdata/structureInputs.txt'

# mimics an eigentriplet from Troy's code in eigentemplate.h
class triplet:
    i = int()
    j = int()
    k = double()

class imrt_class:
    # constants particular to the problem
    numX = 0 # num beamlets
    numvoxels = 0 #num voxels (small voxel space)
    numstructs = 0 # num of structures/regions
    numoars = 0 # num of organs at risk
    numtargets = 0 # num of targets
    numbeams = 0 # num of beams
    totaldijs = 0 # num of nonzeros in Dij matrix

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

    # big D matrix
    Dmat = sparse((1,1), dtype=float) # sparse Dij matrix
    
    # varios folders
    outputDirectory = outputfolder # given by the user in the first lines of *.py
    dataDirectory = readfolder

    # dose variables
    currentDose = [] # dose variable
    currentIntensities = []

    # data class function
    def calcDose(self, newIntensities):
        self.currentIntensities = newIntensities
        self.currentDose = Dmat.transpose * newIntensities
    
    # default constructor
    def __init__(self):
        self.numX = 0

########## END OF CLASS DECLARATION ###########################################
data = imrt_class()

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
bigZ = np.zeros(numVoxels)

# Vorg is a list of the structure voxels in big voxel space
for s in range(0, numStructs):
    Vorg.append(sio.loadmat(allNames[s])['v']-1)
    bigZ[Vorg[s]] = 1.0

# nVox is "small voxel space", with only the voxels that have
# structures assigned (basically non-air/couch voxels)
nVox = sum(bigZ);

# voxelAssignment provides the mapping from small voxel space to big
# voxel space
voxelAssignment = np.zeros(nVox.astype(np.int64))
counter = 0
for i in range(0, numVoxels):
    if(bigZ[i] > 0):
        # If big space voxel is nonzero, save to small vxl space
        voxelAssignment[counter] = i
        counter+=1
print('mapping from small voxel space to big voxel space done')

# originalVoxels is the mapping from big voxel space to small voxel
# space
originalVoxels = np.zeros(numVoxels);
for i in range(0,nVox.astype(np.int64)):
    originalVoxels[voxelAssignment[i].astype(np.int64)] = i

maskValueFull = np.zeros(nVox.astype(np.int64))
maskValueSingle = np.zeros(nVox.astype(np.int64))
# this priority is the order of priority for assigning a single structure per
# voxel (from least to most important)

priority = [7, 24, 25, 23, 22, 21, 20, 16, 15, 14, 13, 12, 10, 11, 9,
4, 3, 1, 2, 17, 18, 19, 5, 6, 8]

for i in range(0, numStructs):
    s = priority[i] - 1
    # generates mask values (the integer that we decompose to get structure
    # assignment). for single it just overrides with the more important
    # structure
    maskValueFull[originalVoxels[Vorg[s]].astype(int)] = maskValueFull[originalVoxels[Vorg[s]].astype(int)]+2**(s-1)
    maskValueSingle[originalVoxels[Vorg[s]].astype(int)] = 2**(s-1);

print('Masking has been calculated')

gastart = 0 ;
gaend = 356;
gastep = degreesep;
castart = 0;
caend = 0;
castep = 0;
ga=[];
ca=[];

os.chdir(readfolderD)
for g in range(gastart, gaend, gastep):
    print(g)
    fname = 'Gantry' + str(g) + '_Couch' + str(0) + '_D.mat'
    bletfname = readfolder + 'Gantry' + str(g) + '_Couch' + str(0) + '_BEAMINFO.mat'
    if os.path.isfile(fname) and os.path.isfile(bletfname):
        ga.append(g)
        ca.append(0)

print('There is enough data for ' + str(len(ga)) + ' different coplanar beam angles\n')

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
beamletCounter = np.zeros(data.numbeams + 1)

for i in range(0, data.numbeams):
    bletfname = readfolder + 'Gantry' + str(ga[i]) + '_Couch' + str(0) + '_BEAMINFO.mat'
    # Get beamlet information
    binfoholder = sio.loadmat(bletfname)

    # Get dose information as in the cpp file
    data.beamNumPerBeam[i] = 10 * int(i) # WILMER. Find out why!!!
    data.beamletsPerBeam[i] = int(binfoholder['numBeamlets'])
    data.dijsPerBeam[i] =  int(binfoholder['numberNonZerosDij'])
    beamletCounter[i+1] = beamletCounter[i] + data.beamletsPerBeam[i]
    
# Generating dose matrix dimensions
data.numX = sum(data.beamletsPerBeam)
data.totaldijs = sum(data.dijsPerBeam)
# Allocate structure for full Dmat file
data.Dmat = sparse.csr_matrix((data.numX, data.numvoxels), dtype=float)

# Work with the D matrices for each beam angle
overallDijsCounter = 0
for i in range(0, data.numbeams):
    fname = 'Gantry' + str(ga[i]) + '_Couch' + str(0) + '_D.mat'
    print('Processing matrix from gantry & couch angle: ' + fname)
    # extract voxel, beamlet indices and dose values
    D = sio.loadmat(fname)['D']
    # write out bixel sorted binary file
    [b,j,d] = sparse.find(D)
    newb = originalVoxels[b]
    
    nBPB[i] = max(j)
    nDIJSPB[i] = len(d)

    # write out voxel sorted binary file
    [jt,bt,dt] = sparse.find(D.transpose())
    newbt = originalVoxels[bt]
    # Notice here that python is smart enough to subtract 1 from matlab's mat
    # files (where the index goes). This was confirmed by Wilmer on 10/19/2015
    tempsparse=sparse.csr_matrix((dt,(jt + beamletCounter[i], newbt)),
                                 shape=(data.numX, data.numvoxels), dtype=float)
    data.Dmat = data.Dmat + tempsparse

    ## Can I erase the below part?
    beamlog = np.ones(len(ga))
    nBeams = len(ga)
    nBeamlets = np.zeros(nBeams)
    rowCumSum = []
print('Finished reading D matrices')

## Read in the objective file:
lines = [myline.split('\t') for myline in [line.rstrip('\n') for line in open(objfile)]]
## Collapse the above expression to a flat list
data.objectiveInputFiles = [item for sublist in lines for item in sublist]
print("Finished reading objective file:\n" + objfile)

## Read in the constraint file:
#####NOTHING TO DO #############

## Read in structures WILMER. CHANGE THIS. Reading from txt file != good!!
lines = [myline.split('\t') for myline in [line.rstrip('\n') for line in open(structurefile)]]
## Collapse the above expression to a flat list
invec = [item for sublist in lines for item in sublist]
## Assignation of different values
maskValueFilename = invec[0]
fullMaskValueFilename = invec[1]
data.numstructs = int(invec[2])
data.numtargets = int(invec[3])
data.numoars = int(invec[4])
# Structure map OARs vs. TARGETs
data.regionIndices = invec[5:(5+data.numstructs)]
data.targets = invec[(5+data.numstructs):(5+2*data.numstructs)]
data.oars = invec[(5+2*data.numstructs):(5+3*(data.numstructs))]
print('Finished reading structures')

# Read in mask values
