import numpy as np

keys = [0,
        15,
        7,22,
        3,18,11,26,
        2,9,5,17,20,28,13,24,
        1,16, 8, 23, 10, 19, 25,27, 4,12,29, 6,14, 21]

for i in keys:
    thislevel = range(i,178, 30)
    a = np.setdiff1d(a, thislevel)
    newentries.append(thislevel)

whoisee = []
for i in range(0,len(newentries)):
    print('level', i, ':')
    for l in newentries[i]:
        print(l)
        whoisee.append(l)
# -----------------------k-medoids analysis functions -------------------------#
print('k-m analysis')
# Create the range of beam angles
ba = range(0, 178)
# Create the set of 16 roughly well spread out beam angles
kappa = []
for i in range(6,178,11):
    kappa.append(i)
print('kappa created', kappa)
from scipy.spatial import distance

def distancemin (kappa, ba):
    for i in ba:
        ba2 = np.matrix([[0,z] for z in ba])
    # Create distance matrix
    D = distance.pdist(ba2)
    D = distance.squareform(D)
    # Select only the rows that are in kappa
    D = D[kappa,:]
    # Find minimum of each column and add up. This is the distance of the whole structure
    dmin = D.min(axis = 0)
    return(sum(dmin))

def listdiff(a, b):
    b = set(b)
    return [aa for aa in a if aa not in b]

def insertnewelement(kappa, ba):
    # Find elements not in kappa
    bcandidates = listdiff(ba, kappa)
    flag = np.inf
    for i in bcandidates:
        kappa.append(i)
        dmi = distancemin(kappa, ba)
        kappa.pop()
        if dmi < flag:
            newbestcandidate = i
            flag = dmi
    kappa.append(newbestcandidate)
    return(kappa)

def givemewholelist(kappa, ba):
    i = 0
    while(len(kappa) < len(ba)):
        kappa = insertnewelement(kappa, ba)
        print('iteration ', i)
        i = i + 1
    return(kappa)
