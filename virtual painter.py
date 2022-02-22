import cv2
import mediapipe as mp
import os
import numpy as np



brushthickness = 25
eraserthickness = 100


folder = "static"
mylist =os.listdir(folder)
overlay = []
for path in mylist:
    image = cv2.imread(f'{folder}/{path}')
    overlay.append(image)
header = overlay[0]
drawcolor = (0,255,0)


cam = cv2.VideoCapture(0)
cam.set(3, 640)
cam.set(4, 720)

mphands = mp.solutions.hands
hands = mphands.Hands()
draw = mp.solutions.drawing_utils
tipid = [4,8,12,16,20]
xp,yp = 0,0
imgcanvas = np.zeros((640,720,3),np.uint8)


while True:
    suc ,img = cam.read()
    img = cv2.flip(img, 1)
    colimg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(colimg)
    lmlist = []
    finguresup = []
    if result.multi_hand_landmarks:
        for handlms in result.multi_hand_landmarks:
            for id , lm in enumerate(handlms.landmark):
                h,w,c = img.shape
                cx , cy = int(lm.x*w), int(lm.y*h)
                lmlist.append([id,cx,cy])
                
        draw.draw_landmarks(img, handlms,mphands.HAND_CONNECTIONS)

    if len(lmlist)!=0:
        
        x1,y1 = lmlist[8][1:]
        x2,y2 = lmlist[12][1:]
        # for thumb
        if lmlist[tipid[0]][1] < lmlist[tipid[0] - 1][1]:
            finguresup.append(1)
        else:
            finguresup.append(0)
        # 4 Fingers
        for id in range(1, 5):
            if lmlist[tipid[id]][2] < lmlist[tipid[id] - 2][2]:
                finguresup.append(1)
            else:
                finguresup.append(0)

    # if two fingers up , slection mode 
    if len(finguresup)!=0:
        if finguresup[1] and finguresup[2]:
            xp,yp = 0,0
            if y1<123:
                if 117<x1<175:
                    header = overlay[0]
                    drawcolor = (0,255,0)
                elif 173<x1<308:
                    header = overlay[1]
                    drawcolor=(0,0,255)
                elif 308 <x1<451:
                    header = overlay[2]
                    drawcolor = (255,0,0)
                elif 451 <x1<640:
                    header = overlay[3]
                    drawcolor = (0,0,0)
            cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawcolor, cv2.FILLED)
    # if one finger is up , drawing mode
        if finguresup[1] and finguresup[2]==False:
            
            cv2.circle(img, (x1, y1), 15, drawcolor, cv2.FILLED)
            if xp==0 and yp==0:
                xp,yp = x1,y1

            if drawcolor == (0,0,0):
                cv2.line(img, (xp, yp), (x1, y1), drawcolor, eraserthickness)
                cv2.line(imgcanvas, (xp, yp), (x1, y1), drawcolor, eraserthickness)
            else:
                cv2.line(img, (xp, yp), (x1, y1), drawcolor, brushthickness)
                cv2.line(imgcanvas, (xp, yp), (x1, y1), drawcolor, brushthickness)
    
            xp,yp = x1,y1

    
   
    img[0:123, 0:640] = header
    
    cv2.imshow("image", img)
    cv2.imshow('canvas', imgcanvas)
    
    cv2.waitKey(1)