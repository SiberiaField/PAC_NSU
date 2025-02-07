import cv2

cap = cv2.VideoCapture("traffic.mp4")
while True:
    ret, frame = cap.read()
    if ret == False:
        break
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow("video", frame)
    key = cv2.waitKey(20)
    if key == 27:
        break
cv2.destroyAllWindows()
cap.release()