
import glob, os
import numpy as np
import scipy.io as sio
from scipy import sparse

# First of all make sure that I can read the data


# In the data directory with the *VOILIST.mat files, this opens up
# each structure file and reads in the structure names and sizes in
# voxels

readfolder = '/home/wilmer/Documents/Troy_BU/Data/DataProject/HN/'
readfolderD = '/home/wilmer/Documents/Troy_BU/Data/DataProject/HN/Dij/'
outputfolder = '/home/wilmer/Dropbox/Research/VMAT/output/'

oldfolder = os.getcwd()
os.chdir(readfolder)
allFiles = glob.glob("*VOILIST.mat")
allNames = sorted(allFiles) #Make sure it's sorted because it was not.
numStructs = len(allFiles)
print(numStructs)

# This is "big voxel space" where some voxels may receive no dose or
# have no structure assigned # Wilmer. Find where this comes from
numVoxels = 1715200

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
        voxelAssignment[counter] = i
        counter+=1

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

os.chdir('/home/wilmer/Documents/Troy_BU/Data/DataProject/HN/Dij')
gastart = 0 ;
gaend = 356;
gastep = 60;
castart = 0;
caend = 0;
castep = 0;
ga=[];
ca=[];

for g in range(gastart, gaend, gastep):
    fname = 'Gantry' + str(g) + '_Couch' + str(0) + '_D.mat'
    if os.path.isfile(fname):
        ga.append(g)
        ca.append(0)

# build new sparse matrices
        
# This code translates the sparse dose matrices from big voxel space to
# small voxel space and writes it out to a binary file to be used in the
# optimization

# nBPB is num beamlets per beam
nBPB = np.zeros(len(ga))
# nDIJSPB is the number of nonzeros in the Dmatrix for each beam
nDIJSPB = np.zeros(len(ga))

for i in range(0, len(ga)):
    fname = 'Gantry' + str(ga[i]) + '_Couch' + str(0) + '_D.mat'
    print(fname)
    print(fname + '   ' + ' of ' + str(len(ga)))
    # extract voxel, beamlet indices and dose values
    D = sio.loadmat(fname)['D']
    [b,j,d] = sparse.find(D)

    nBPB[i] = max(j)
    nDIJSPB[i] = len(d)
    newb = originalVoxels[b]
    
    # write out voxel sorted binary file
    [jt,bt,dt] = sparse.find(D.transpose())
    newbt = originalVoxels[bt]
    
nBPB = nBPB.astype(np.int64)
nDIJBPB = nDIJBPB.astype(np.int64)
    
beamlog = np.ones(len(ga))
nBeams = len(ga)
beamlog(nBeams)
nBeamlets = np.zeros(nBeams)
rowCumSum = []

# Column generation based greedy heuristic for Master Problem

# Ccalig contains the list of the apertures that enter as nonzero. I start with
# an empty list.
Ccalig = []
zbar = 0
K = range(0, 360, 2)

# for each aperture not in CCALIG do the pricing problem subroutine
for i in [x for x in K if x not in Ccalig]:
    print(i)
