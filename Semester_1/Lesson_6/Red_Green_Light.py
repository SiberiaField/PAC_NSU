import cv2
import time

def getMask(fg_mask):
    blured_mask = cv2.GaussianBlur(fg_mask, (3, 3), 3)
    _, mask_thresh = cv2.threshold(blured_mask, 180, 255, cv2.THRESH_BINARY)
    # вычисление ядра
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # Apply erosion
    mask_eroded = cv2.morphologyEx(mask_thresh, cv2.MORPH_OPEN, kernel)
    return mask_eroded

def getContours(mask_eroded):
    contours, _ = cv2.findContours(mask_eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_contour_area = 1000  # Define your minimum area threshold
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]
    return contours

def getRectangles(movements_contours, frame_out):
    for cnt in movements_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        frame_out = cv2.rectangle(frame_out, (x, y), (x+w, y+h), (0, 0, 255), 3)

backSub = cv2.createBackgroundSubtractorMOG2()
cap = cv2.VideoCapture(0)
timing = time.time()
detecting = True
while True:
    ret, frame = cap.read()
    if ret == False:
        break

    change = time.time() - timing >= 3
    if change:
        detecting = not detecting
        timing = time.time()

    fg_mask = backSub.apply(frame)
    frame_out = frame.copy()
    if detecting:
        h, w = frame_out.shape[:2]
        frame_out = cv2.rectangle(frame_out, (0, 0), (w - 1, h - 1), (0, 255, 0), cv2.FILLED)
        movements_mask = getMask(fg_mask)
        movements_contours = getContours(movements_mask)
        #getRectangles(movements_contours, frame)
        frame_out = cv2.drawContours(frame_out, movements_contours, -1, (0, 0, 255), cv2.FILLED)

    if detecting:
        cv2.putText(frame_out, 'Red light', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        cv2.putText(frame_out, 'Green light', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('cam', frame_out)
    key = cv2.waitKey(20)
    if key == 27:
        break
    
cv2.destroyAllWindows()
cap.release()