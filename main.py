import numpy as np
from PIL import Image

pixX = 250
pixY = 250

setSize = 100

def img_mean(img, mean):
    numpydata = np.asarray(img)
    mean = np.add(mean, numpydata)
    return mean


def main():
    mean = np.zeros((pixX,pixY))

    #Getting mean
    for i in range(setSize):
        name = 'train/cat.' + str(i) + '.jpg'
        img = Image.open(name).convert('L')
        img = img.resize((pixX,pixY))
        mean = img_mean(img, mean)
    mean = mean.dot(1/setSize)

    #Getting covMatrix
    sum = np.zeros((pixX,pixY))
    for i in range(setSize):
        name = 'train/cat.' + str(i) + '.jpg'
        img = Image.open(name).convert('L')
        img = img.resize((pixX,pixY))
        data = np.asarray(img)
        data = np.subtract(data, mean)
        sum = np.add(sum, data.dot(data.transpose()))
    covMatrix = sum.dot(1/setSize)

    #Eigen value stuff
    eigen = np.linalg.eig(covMatrix)
    sorted = np.sort(eigen[0])
    index_arr = [np.where(eigen[0] == sorted[i]) for i in range(len(eigen[0]))]

    #Get sorted eigen vectors
    new = swap_indexes(eigen[1], index_arr)
    new = np.array(new)
    new = new.dot(255).astype('uint8')
    img = Image.fromarray(new)
    img.convert('L')
    img.show()

    return 0

def swap_indexes(arr, indexes):
    ret = []
    for i in range(len(indexes)):
        ret.append(arr[indexes[i][0]][0])
    return ret

if __name__ == "__main__":
    main()