from csv import DictReader
import numpy as np
import math
from scipy.cluster.hierarchy import dendrogram as den
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage as link
import time

def mutData(matrix, index1, index2, i):
    nClust = math.floor(matrix[index1, index1]/bignum) + math.floor(matrix[index2,index2]/bignum)
    index1 = int(index1)
    index2 = int(index2)
    matrix[index1] = np.max(np.vstack((matrix[index1], matrix[index2])), axis=0)
    matrix[:,index1] = np.transpose(matrix[index1])
    matrix[index2,:] = bignum
    matrix[:,index2] = bignum
    matrix[index1,index1] = bignum * nClust + i + len(matrix)
    return matrix

def load_data(filepath):
    file = open(filepath, "r")
    data = list(DictReader(file))
    for x in data:
        dict(x)
    return data

def calc_features(row):
    ft = [int(row["Attack"]), int(row["Sp. Atk"]), int(row["Speed"])
          , int(row["Defense"]), int(row["Sp. Def"]), int(row["HP"])]
    return np.array(ft, dtype="int64")

def hac(features):
    global bignum
    bignum = 1000000
    n = len(features)
    if (n > bignum):
        bignum = 10**(math.floor(math.log(n), 10)+3)
    
    x = np.asmatrix(features)
    gram = x * np.transpose(x)
    mones = np.ones((1, n))
    x2 = np.square(x)
    c1 = x2.sum(axis=1)
    c2 = np.ravel(c1)
    dMat = np.sqrt((c1*mones) + (c2*mones) - (2*gram))
    for i in range(n):
        dMat[i, i] = bignum + i
    
    output = np.zeros((n-1, 4))
    for i in range(n-1):
        smallest = np.array(np.where(dMat == dMat.min()))
        ism = []
        for x in smallest[0]:
            ism.append(dMat[x,x])
        ind = smallest[:,np.argmin(ism)]
        dval = dMat[ind[0], ind[1]]
        i1 = min(ind[0], ind[1])
        i2 = max(ind[0], ind[1])
        ci1, ci2 = i1, i2
        i1 = int(dMat[i1, i1] % bignum)
        i2 = int(dMat[i2, i2] % bignum)
        
        dMat = mutData(dMat, ci1, ci2, i)

        output[i, 0] = min(i1, i2)
        output[i, 1] = max(i1, i2)
        output[i, 2] = dval
        output[i, 3] = math.floor(dMat[ci1,ci1]/bignum)
    return output

def imshow_hac(Z, names):
    plt.subplot()
    plt.tight_layout()
    den(Z, labels=names, leaf_rotation=90)
    plt.show()

featandnames = [(calc_features(row), row['Name']) for row in load_data('Pokemon.csv')[:100]]
stime = time.time()
Z = hac([row[0] for row in featandnames])
etime = time.time()
lk = link([row[0] for row in featandnames], method="complete")
e2time = time.time()
#names = [row[1] for row in featandnames]
#imshow_hac(lk, names)
#print(Z)
#print()
#print(lk)
#print(np.array_equal(Z, lk))
print("hac time: " + str(etime - stime))
print("link time: " + str(e2time - etime))