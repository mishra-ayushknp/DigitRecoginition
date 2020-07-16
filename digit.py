#importing library
import matplotlib.pyplot as plt
from sklearn import datasets, svm
import cv2
from imageio import imread 
#loading datasets from MNIST datasets
digit = datasets.load_digits()
#calculating length of sample
n_sa = len(digit.images)
#Reshaping into [1,1] matrix
imd = digit.images.reshape((n_sa,-1))
#classifier using Support Vector Machine
classifier = svm.SVC(gamma = 0.001)
#fitting the  data for prediction of one half of the sample
classifier.fit(imd[:n_sa//2],digit.target[:n_sa//2])
#predicting with another half dataset
classifier.predict(imd[n_sa//2:])
# loading the image file
img = imread("Seven.jpeg")
#resizing the image into (8,8) as same as the size of the datasets image
gray = cv2.resize(img,(8,8),interpolation = cv2.INTER_AREA)

i = cv2.normalize(src = gray , dst = None , alpha = 0 , beta = 16 , norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_8U)

classifier = svm.SVC(gamma = 0.001)
classifier.fit(imd[:],digit.target[:])
xx = []
for r in i :
    for c in r :
        xx.append(sum(c)/3.0)
prww = classifier.predict([xx])
print(img.dtype)
print(prww)
#print(prww)
#print(gray)
