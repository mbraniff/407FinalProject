import numpy as np
from PIL import Image
from sklearn.decomposition import PCA

def main():
    img = Image.open('train/cat.0.jpg').convert('LA')
    numpydata = np.asarray(img)
    x = len(numpydata)
    y = len(numpydata[0])
    mean = 0
    for i in range(x):
        for j in range(y):
            mean += numpydata[i][j][0]
    mean /= len(numpydata)*len(numpydata[0])
    arr = []
    for i in range(x):
        arr.append([])
        for j in range(y):
            grey = numpydata[i][j][0] - mean
            arr[i].append(grey)
    arr = np.array(arr)
    covMatrix = arr.dot(arr.transpose())
    eigen = np.linalg.eig(covMatrix)
    index = np.where(eigen[0] == np.amax(eigen[0]))
    eigVec = eigen[1][index]
    print(eigVec.shape)
    return 0


if __name__ == "__main__":
    main()