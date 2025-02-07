import cv2
import os

pairs = []
for _, _, fnames in os.walk(os.path.abspath("nails_segmentation\\images")):
    for name in fnames:
       img = cv2.imread("nails_segmentation\\images\\" + name)
       lbl = cv2.imread("nails_segmentation\\labels\\" + name)
       pairs.append([img, lbl])

def showPairs():
    for pair in pairs:
        cv2.imshow("image", pair[0])
        cv2.imshow("label", pair[1])
        key = cv2.waitKey(0)
        if key == 27:
            break
    cv2.destroyAllWindows()

def showContours():
    for pair in pairs:
        label = cv2.cvtColor(pair[1], cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(label, 200, 255, cv2.THRESH_BINARY)
        contour, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        res = cv2.drawContours(pair[0], contour, -1, (0, 255, 0), 2)
        cv2.imshow("imgWithContour", res)
        key = cv2.waitKey(0)
        if key == 27:
            break
    cv2.destroyAllWindows()

showPairs()
showContours()