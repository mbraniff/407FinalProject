import numpy as np
from PIL import Image

pixX = 250
pixY = 250

SETSIZE = 50

class EigenFaces:
    def __init__(self):
        self.faces = np.array([])
        self.count = 0
        self.mean = np.array([])

    def add_face(self, face):
        self.count += 1
        self.faces = np.append(self.faces, face)

    def add_faces(self, faceArr, selectedArr):
        for index in selectedArr:
            self.faces = np.append(self.faces, faceArr[index])
            self.count += 1

    def done(self):
        self.faces = self.faces.reshape((self.count, pixX*pixY,))
        for i in range(self.count):
            self.mean = np.append(self.mean, self.faces[i].sum()/(pixX*pixY))
        self.mean = self.mean.dot(1/self.count)

    def test_face(self, face):
        distance = 0
        for i in range(self.count):
            alpha = face.transpose().dot(self.faces[i])
            distance += pow(self.mean[i] - alpha, 2)
        return distance

def getMaxEigenVector(dir, index, setSize):
    #Getting covMatrix
    X = []
    for i in range(index, index + setSize):
        name = dir + str(i) + '.jpg'
        img = Image.open(name).convert('L')
        img = img.resize((pixX, pixY))
        data = np.asarray(img)
        data = data.ravel()
        X.append(data)
    X = np.array(X)
    X = X.transpose()

    #Eigen value stuff
    L = X.transpose().dot(X)
    eigen = np.linalg.eig(L)
    sorted = np.sort(eigen[0])
    index_arr = [np.where(eigen[0] == sorted[i]) for i in range(len(eigen[0]))]

    #Get sorted eigen vectors
    eigenVectors = np.array(swap_indexes(eigen[1], index_arr))
    ret = np.array([])
    for i in range(setSize):
        ret = np.append(ret, X.dot(eigenVectors[i]))
    return ret

def displayEigenFace(face):
    reravel = face.reshape(pixX, pixY)
    img = Image.fromarray(reravel)
    img = img.convert('L')
    img.show()

def main():
    if 12500%SETSIZE != 0:
        print("Set size must evenly divide into 12500")
        return 0
    cats = EigenFaces()
    dogs = EigenFaces()
    catTest = getMaxEigenVector('train1/cat.', 0, SETSIZE)
    dogTest = getMaxEigenVector('train1/dog.', 0, SETSIZE)

    catTest = catTest.reshape((SETSIZE, pixX * pixY,))
    dogTest = dogTest.reshape((SETSIZE, pixX * pixY,))

    cats.add_faces(catTest, [39, 46, 36, 34, 32, 31, 30, 29, 28, 27, 26, 25, 21, 5, 6])
    dogs.add_faces(dogTest, [3, 5, 7, 9, 10, 12, 16, 17, 18, 22, 23, 24, 25, 26, 44])

    cats.done()
    dogs.done()
    user = input("Face to compare, # to exit\n")
    while(user != "#"):
        index = int(user)
        path = "test2/" + str(index) + ".jpg"
        img = Image.open(path).convert('L').resize((pixX,pixY))
        test = np.asarray(img).ravel()
        catDistance = cats.test_face(test)
        dogDistance = dogs.test_face(test)
        print(catDistance, dogDistance)
        if catDistance > dogDistance:
            print("Image is a dog")
        else:
            print("Image is a cat")
        user = input("Face to compare, # to exit\n")
    return 0

def swap_indexes(arr, indexes):
    ret = []
    for i in range(len(indexes)):
        ret.append(arr[indexes[i][0]][0])
    return ret

if __name__ == "__main__":
    main()