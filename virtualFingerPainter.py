import cv2
import mediapipe as mp
import time
import numpy as np


def findPosition(img, handNo=0, draw=True):
    lmList = []
    if results.multi_hand_landmarks:
        myHand = results.multi_hand_landmarks[handNo]
        for id, lm in enumerate(myHand.landmark):
            # print(id, lm)
            h, w, c = img.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            # print(cx, cy)
            lmList.append([id, cx, cy])
            if (draw):
                cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
    return lmList

def fingersUp():
    fingers = []
    tipIds = [4, 8, 12, 16, 20]
    if(lmList[tipIds[0]][1] < lmList[tipIds[0] - 1][1]):
        fingers.append(1)
    else:
        fingers.append(0)
    for id in range(1, 5):
        if(lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]):
            fingers.append(1)
        else:
            fingers.append(0)
    return fingers



cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

drawColor = (125,50,60)
brushThickness = 15
xp, yp = 0, 0
imgCanvas = np.zeros((720, 1280, 3), np.uint8)

mpHands = mp.solutions.hands
hands = mpHands.Hands(min_detection_confidence=0.85)
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

while (True):
    success, img = cap.read()
    img = cv2.flip(img, 1)

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    cTime = time.time()
    fbs = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fbs)), (10, 70), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 0, 255), 3)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    lmList = findPosition(img, draw=False)
    if len(lmList) != 0:

        fingers = fingersUp()
        # print(fingers)

        if fingers[1] and fingers[2]:
            xp, yp = 0, 0
            drawColor = (125,50,60)
            brushThickness = 15

        elif fingers[4] and (fingers[1] and fingers[2] and fingers[3] and fingers[0]) == False:
                drawColor = (0, 0, 0)
                brushThickness = 50
                x1, y1 = lmList[4][1:]
                cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)
                print("Erasing Mode")
                if xp == 0 and yp == 0:
                    xp, yp = x1, y1
                cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)
                xp, yp = x1, y1

        elif fingers[1] and fingers[2] == False:
            x1, y1 = lmList[8][1:]
            drawColor = (125, 50, 60)
            brushThickness = 15
            cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)
            print("Drawing Mode")
            if xp == 0 and yp == 0:
                xp, yp = x1, y1
            cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
            cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)
            xp, yp = x1, y1

        elif (fingers[0] and fingers[1] and fingers[2] and fingers[3] and fingers[4]) == False:
        #     # xp, yp = 0, 0
            imgCanvas = np.zeros((720, 1280, 3), np.uint8)


    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    display = cv2.bitwise_and(img, imgInv)
    display = cv2.bitwise_or(display, imgCanvas)
    # finalImg = cv2.addWeighted(img, 0.5, imgCanvas, 0.5, 0)
    cv2.imshow("image", display)
    # cv2.imshow("Canvas", imgCanvas)
    cv2.waitKey(1)


