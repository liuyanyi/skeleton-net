import cv2
import time
import numpy as np

cap = cv2.VideoCapture(0)
i=0
start = time.time()
while(1):
    # get a frame
    ret, frame = cap.read()
    # show a frame
    cv2.imshow("capture", frame)
    i = i+1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    nt = time.time()-start
    print(i/nt)
cap.release()
cv2.destroyAllWindows() 