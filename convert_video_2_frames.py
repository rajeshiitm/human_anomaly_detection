


import sys
import argparse

import cv2
print(cv2.__version__)

def extractImages(pathIn, pathOut):
    """
    We do not extract every frame but wants to extract frame every one second. 
    So a 1-minute video will give 60 frames(images).
    """
    count = 0
    vidcap = cv2.VideoCapture(pathIn)
    success,image = vidcap.read()
    success = True
    while success:
         # added this line 
        vidcap.set(cv2.CAP_PROP_POS_MSEC,(count*1000))   
        success,image = vidcap.read()
        print ('Read a new frame: ', success)
        # save frame as JPEG file
        cv2.imwrite( pathOut + "\\frame%d.jpg" % count, image)     
        count = count + 1

if __name__=="__main__":
    pathIn = "E:\Hack-O-Pitch_Human_anamoly\stealing.mp4"
    pathOut = "E:\Hack-O-Pitch_Human_anamoly\stealing"

    extractImages(pathIn, pathOut)