import glob, os
import numpy as np
import scipy.io as sio

# In the data directory with the *VOILIST.mat files, this opens up
# each structure file and reads in the structure names and sizes in
# voxels

oldfolder = os.getcwd()
os.chdir('/run/media/wilmer/DSOL_BU/Troy_BU/Data/DataProject/HN')
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
    Vorg.append(sio.loadmat(allNames[s])['v'])
    bigZ[Vorg[s]-1] = 1.0

# nVox is "small voxel space", with only the voxels that have
# structures assigned (basically non-air/couch voxels)
nVox = sum(bigZ);

# voxelAssignment provides the mapping from small voxel space to big
# voxel space
voxelAssignment = np.zeros(nVox)
counter = 0
for i in range(0, numVoxels):
    if(bigZ[i] > 0):
        voxelAssignment[counter] = i
        counter+=1

# originalVoxels is the mapping from big voxel space to small voxel
# space
originalVoxels = np.zeros(numVoxels);
for i in range(0,nVox.astype(np.int64)):
    originalVoxels[voxelAssignment[i]] = i

maskValueFull = np.zeros(nVox)
maskValueSingle = np.zeros(nVox)

# this priority is the order of priority for assigning a single structure per
# voxel (from least to most important)

priority = [7, 24, 25, 23, 22, 21, 20, 16, 15, 14, 13, 12, 10, 11, 9,
 4, 3, 1, 2, 17, 18, 19, 5, 6, 8]

for i in range(0, numStructs):
    s = priority[i]
    # generates mask values (the integer that we decompose to get structure
    # assignment). for single it just overrides with the more important
    # structure
    maskValueFull[originalVoxels[Vorg[s]]] =
    maskValueFull(originalVoxels(Vorg{s}))+2^(s-1)
    maskValueSingle(originalVoxels(Vorg{s})) = 2^(s-1);

os.chdir('/run/media/wilmer/DSOL_BU/Troy_BU/Data/DataProject/HN/Dij')
gastart = 0 ;
gaend = 356;
gastep = 60;
castart = -90;
caend = 90;
castep = 15;
ga=[];
ca=[];

for g in range(gastart, gaend, gastep):
    for c in range(castart, caend, castep):
        fname = 'Gantry' + str(g) + '_Couch' + str(c) + '_D.mat'
        if os.path.isfile(fname):
            ga.append(g)
            ca.append(c)

