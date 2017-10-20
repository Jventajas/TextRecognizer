import cv2
import numpy as np
import operator
import os
import matplotlib.pyplot as plt
import pandas as pd

import os
import pytesseract
from scipy import misc
from PIL import Image

# module level variables ##########################################################################
MIN_CONTOUR_AREA = 100
RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 30

###################################################################################################
class ContourWithData():

    # member variables ############################################################################
    npaContour = None           # contour
    boundingRect = None         # bounding rect for contour
    intRectX = 0                # bounding rect top left corner x location
    intRectY = 0                # bounding rect top left corner y location
    intRectWidth = 0            # bounding rect width
    intRectHeight = 0           # bounding rect height
    fltArea = 0.0               # area of contour

    def calculateRectTopLeftPointAndWidthAndHeight(self):               # calculate bounding rect info
        [intX, intY, intWidth, intHeight] = self.boundingRect
        self.intRectX = intX
        self.intRectY = intY
        self.intRectWidth = intWidth
        self.intRectHeight = intHeight

    def checkIfContourIsValid(self):                            # this is oversimplified, for a production grade program
        if self.fltArea < MIN_CONTOUR_AREA: return False        # much better validity checking would be necessary
        return True






### trying to detect VIN of an image

allContoursWithData = [] # declare empty lists,
validContoursWithData = [] # we will fill these shortly

# read labels of training images
npaClassifications = np.loadtxt("classifications.txt", np.float32)
# read training images
npaFlattenedImages = np.loadtxt("flattened_images.txt", np.float32)

# original: reshape numpy array to 1d, necessary to pass to call to train
npaClassifications = npaClassifications.reshape((npaClassifications.size, 1))


# instantiate KNN object
kNearest = cv2.KNearest()
kNearest.train(npaFlattenedImages, npaClassifications)


# load image from file
imgToClassify = cv2.imread("AnylinVINScans/IMG_0407.JPG")
# preprocess image for finding contours & classification
imgThresh = preprocess_image(imgToClassify)

# make a copy of the thresh image, this in necessary b/c findContours modifies the image
imgThreshCopy = imgThresh.copy()



# arg1: input image, make sure to use a copy since the function will modify this image in the course of finding contours
# arg2: retrieve the outermost contours only
# arg3: compress horizontal, vertical, and diagonal segments and leave only their end points
npaContours, npaHierarchy = cv2.findContours(imgThreshCopy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


allContoursWithData = [] # declare empty lists,
validContoursWithData = [] # we will fill these shortly








### loop over all contour

# for each contour
for npaContour in npaContours:
    # instantiate a contour with data object
    contourWithData = ContourWithData()
    # assign contour to contour with data                               
    contourWithData.npaContour = npaContour
    # get the bounding rect
    contourWithData.boundingRect = cv2.boundingRect(contourWithData.npaContour)
    # get bounding rect info
    contourWithData.calculateRectTopLeftPointAndWidthAndHeight()
    # calculate the contour area
    contourWithData.fltArea = cv2.contourArea(contourWithData.npaContour)
    # add contour with data object to list of all contours with data
    allContoursWithData.append(contourWithData)


# for all contours
for contourWithData in allContoursWithData:
    if contourWithData.checkIfContourIsValid():
        # if valid, append to valid contour list
        validContoursWithData.append(contourWithData)






# sort contours from left to right
validContoursWithData.sort(key = operator.attrgetter("intRectX"))
# declare final string for the final number sequence by the end
strFinalString = ""

# for each contour
for contourWithData in validContoursWithData:
    # draw a green rect around the current char
    
    # arg1: draw rectangle on original testing image
    # arg2: upper left corner
    # arg3: lower right corner
    # arg4: green
    # arg5: thickness
    cv2.rectangle(imgToClassify,
                  (contourWithData.intRectX, contourWithData.intRectY),
                  (contourWithData.intRectX + contourWithData.intRectWidth, contourWithData.intRectY + contourWithData.intRectHeight),
                  (0, 255, 0),
                  2)
    # crop char out of threshold image
    imgROI = imgThresh[contourWithData.intRectY : contourWithData.intRectY + contourWithData.intRectHeight,
                       contourWithData.intRectX : contourWithData.intRectX + contourWithData.intRectWidth]
    # resize image, this will be more consistent for recognition and storage
    imgROIResized = cv2.resize(imgROI, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))
    # flatten image into 1d numpy array
    npaROIResized = imgROIResized.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))
    # convert from 1d numpy array of ints to 1d numpy array of floats
    npaROIResized = np.float32(npaROIResized)
    # call KNN function find_nearest
    retval, npaResults, neigh_resp, dists = kNearest.find_nearest(npaROIResized, k = 1)
    # get character from results
    strCurrentChar = str(chr(int(npaResults[0][0])))
    # append current char to full string
    strFinalString = strFinalString + strCurrentChar

