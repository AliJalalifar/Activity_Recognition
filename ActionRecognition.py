from glob import glob
from os import listdir
from os.path import isfile, join
import os
import numpy as np
import cv2
from skimage import measure
from sklearn.externals import joblib
import re

#Find mode in an array

def modefinder(array):
    one = np.count_nonzero(array == 1)
    two = np.count_nonzero(array == 2)
    three = np.count_nonzero(array == 3)
    four = np.count_nonzero(array == 4)
    five = np.count_nonzero(array == 5)
    max = np.argmax([0,one,two,three,four,five])
    return max

# crop white part
def crop_image(img,tol=0):
    mask = img>tol
    return img[np.ix_(mask.any(1),mask.any(0))]

clf1 = joblib.load("model/mlp.model")

Activities = [0,'Waving 1 hand','Waving 2 hands','Jogging in place','Jumping in place','Picking up']
counter = 0;
cv2.ocl.setUseOpenCL(False)
paths = glob('Tests\\*\\*\\')
kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9))
height,width = 720,960

#Loading Backgrounds

view_backgrounds = []
for img in glob('Background\\*.jpg'):
    n = cv2.imread(img)
    view_backgrounds.append(n)

#Saving To folders
print "Saving Background of each view in corrosponding folder!"

for j in range(0,len(paths)):
    for i in range(1,9):
        if paths[j].__contains__("view" + str(i)):
            cv2.imwrite(paths[j]+'\\0.jpg', view_backgrounds[i-1])
print "Done!"

#Silhouette Extraction

for  p in paths:
    print "Silhoette Extraction. Progress  -> " + str(counter*1.0/len(paths)*100) + "%"
    counter = counter+1
    mypath = str(p)
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    images = np.empty(len(onlyfiles), dtype=object)
    for n in range(0, len(onlyfiles)):
        images[int(str(onlyfiles[n])[:-4])] = cv2.imread(join(mypath, onlyfiles[n]), 0)
    fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
    for i in range(0, len(images)):
        fgmask = fgbg.apply(images[i], learningRate=0.000001)
        opening = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel2)
        im2, contours, hierarchy = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE, )
        cv2.drawContours(im2, contours, -1, (255, 255, 255), thickness=3, maxLevel=2)
        cv2.fillPoly(im2, contours, (255, 255, 255))
        if not os.path.exists('TempSilhouette\\' + p[6:]):
            os.makedirs('TempSilhouette\\' + p[6:])
        if(i>0):
            cv2.imwrite('TempSilhouette\\' + p[6:] + "\\" + str(i) + '.jpg', im2)

print "Done!"


#Create Motion History Image

motionpaths = glob('TempSilhouette\\*\\*\\')

myimage = []
counter = 0
foldercounter =0;
for j in range(0,len(motionpaths)):
    print "Create Motion History Image. Progress  -> " + str(foldercounter*1.0/len(motionpaths)*100) + "%"
    foldercounter = foldercounter +1
    motionkernel = np.ones((5, 5), np.uint8)
    for view in range(0, 8):
        if motionpaths[j].__contains__("view" + str(view+1)):
            counter = 0;
            myimage = []
            for img in sorted(glob(motionpaths[j] + '\\*.jpg'),key=os.path.getmtime):
                if(counter<5):
                    if(counter==0):
                        firstimage = cv2.imread(img)
                    myimage.append(cv2.imread(img))
                else:
                    myimage.append(cv2.imread(img))
                    MotionImage = cv2.bitwise_or( cv2.bitwise_or( cv2.bitwise_or(myimage[counter],myimage[counter-1]), cv2.bitwise_or(myimage[counter-2],myimage[counter-3])),cv2.bitwise_or(myimage[counter-4],myimage[counter-5]))
                    MotionImage = cv2.subtract(MotionImage,firstimage)
                    MotionImage = cv2.erode(MotionImage,motionkernel)
                    if not os.path.exists('MotionImage\\' + motionpaths[j][15:]):
                        os.makedirs('MotionImage\\' + motionpaths[j][15:])
                    cv2.imwrite('MotionImage\\' + motionpaths[j][15:] + "\\" + str(counter-4) + '.jpg', MotionImage)
                counter = counter+1;

print "Done!"


# Hog Initialization

winSize = (64,64)
blockSize = (16,16)
blockStride = (8,8)
cellSize = (16,16)
nbins = 9
derivAperture = 1
winSigma = 4.
histogramNormType = 0
L2HysThreshold = 2.0000000000000001e-01
gammaCorrection = 0
nlevels = 64
hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,
                        histogramNormType,L2HysThreshold,gammaCorrection,nlevels)
winStride = (64,64)
padding = (8,8)
locations = ((0,0),)

TestSet = glob('MotionImage\\*\\')
imagepath = []
perdictions = [[]]
labelcounting = 0;
for testPath in TestSet:
    imagepath.append(sorted(glob(testPath+"*\\*.jpg"),key=os.path.getmtime))


for testSize in range(0,len(imagepath)):
    perdictions = [[]]
    print "Labling in Progress  -> " + str(labelcounting * 1.0 / len(imagepath) * 100) + "%"
    labelcounting = labelcounting + 1;
    file = open("Tests" + os.path.dirname(os.path.dirname(imagepath[testSize][0]))[11:] + "\\labels.txt", "a")
    cpt = [len(files) for r, d, files in os.walk(os.path.dirname(os.path.dirname(imagepath[testSize][0])))]
    for i in range (1,np.max(cpt)+1):
        matches = filter(lambda s: "\\"+str(i)+".jpg" in str(s), imagepath[testSize])
        labelarray = [0, 0, 0, 0, 0, 0, 0, 0]
        probabilityarray = [0, 0, 0, 0, 0, 0, 0, 0]
        for j in range(0,len(matches)):
            images = cv2.imread(matches[j],0)
            white = np.count_nonzero(images)
            if(white>1000):
                cropped = crop_image(images,0)
                hist = hog.compute(cropped, winStride, padding, locations)
                features = np.reshape(np.asarray(hist), (441,))
                label = clf1.predict(features)
                labelarray[j] = label
                probabilityarray[j] = max(max(clf1.predict_proba(features)))
        labelnumbers = [int(s) for s in re.findall(r'\b\d+\b', str(labelarray))]
        perdictions.append(labelnumbers[np.argmax(probabilityarray)])
        perdiction = labelnumbers[np.argmax(probabilityarray)]
        file.write("Frame #" + str(i+5)+ " = " + str(Activities[perdiction]) + "\n")
