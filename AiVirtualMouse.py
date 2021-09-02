import cv2
import numpy as np
import time
import os
import HandTrackingModule as htm
######

brushThickness=10
eraserThickness=50
####


folderPath="Resources"
myList=os.listdir(folderPath)
print(myList)
overlay=[]
for imgPath in myList:
    image=cv2.imread(f'{folderPath}/{imgPath}')
    overlay.append(image)

header=overlay[0]
drawColor=(230,108,203)
cap=cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)
detector=htm.HandDetector(minDetect=0.85)
xp,yp=0,0
imgCanvas=np.zeros((720,1280,3),np.uint8)
while True:


    #1. Import the camera footage
    sucess,img=cap.read()
    img=cv2.flip(img,1)

    #2. Detect Hand Landmarks
    img=detector.findHands(img)
    lmList=detector.findPosition(img,draw=False)

    if len(lmList) !=0:

        x1, y1=lmList[8][1:]
        x2, y2 = lmList[12][1:]

        fingers=detector.fingersUp()
        # print(fingers)
        # If selection mode
        if fingers[1] and fingers[2]:
            xp, yp = 0, 0
            print("Selection mode")
            if y1<125:
                if 188<x1<330:
                    header=overlay[0]
                    drawColor=(230,108,203)
                if 454<x1<594:
                    header=overlay[1]
                    drawColor=(77, 145, 255)
                if 716<x1<860:
                    header=overlay[2]
                    drawColor=(173, 74,0)
                if 946<x1<1094:
                    header=overlay[3]
                    drawColor=(0,0,0)
            cv2.rectangle(img,(x1,y1-25),(x2,y2+25),drawColor,cv2.FILLED)



        #If drawing mode
        if fingers[1] and fingers[2]==False:
            cv2.circle(img,(x1,y1),10,drawColor,cv2.FILLED)
            print("Drawing mode")

            if xp==0 and yp==0:
                xp,yp=x1,y1

            if drawColor ==(0,0,0):
                cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)

            cv2.line(img,(xp,yp),(x1,y1),drawColor,brushThickness)
            cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)
            xp, yp=x1, y1


    imgGray=cv2.cvtColor(imgCanvas,cv2.COLOR_BGR2GRAY)
    t, imgInv=cv2.threshold(imgGray,50,255,cv2.THRESH_BINARY_INV)
    imgInv=cv2.cvtColor(imgInv,cv2.COLOR_GRAY2BGR)
    img=cv2.bitwise_and(img,imgInv)
    img=cv2.bitwise_or(img,imgCanvas)



    #Setting the header image
    img[0:125,0:1280]=header
    cv2.imshow("Output",img)
    #cv2.imshow("OutputCanvas", imgCanvas)
    cv2.waitKey(1)