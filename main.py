import numpy as np
from PIL import Image

def main():
    img = Image.open('train/cat.0.jpg').convert('L')
    img = img.resize((250,250))
    img.show()
    numpydata = np.asarray(img)
    x = len(numpydata)
    y = len(numpydata[0])
    mean = 0
    for i in range(x):
        for j in range(y):
            mean += numpydata[i][j]
    mean /= len(numpydata)*len(numpydata[0])
    arr = []
    for i in range(x):
        arr.append([])
        for j in range(y):
            grey = numpydata[i][j] - mean
            arr[i].append(grey)
    arr = np.array(arr)
    covMatrix = arr.dot(arr.transpose())
    eigen = np.linalg.eig(covMatrix)
    sorted = np.sort(eigen[0])
    index_arr = [np.where(eigen[0] == sorted[i]) for i in range(len(eigen[0]))]
    new = swap_indexes(eigen[1], index_arr)
    new = np.array(new)
    new = new.dot(255).astype('uint8')
    img = Image.fromarray(new)
    img.convert("L")
    img.show()
    return 0

def swap_indexes(arr, indexes):
    ret = []
    for i in range(len(indexes)):
        ret.append(arr[indexes[i][0]][0])
    return ret

if __name__ == "__main__":
    main()