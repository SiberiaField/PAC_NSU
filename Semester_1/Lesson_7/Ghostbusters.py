import cv2
import numpy as np

sift = cv2.SIFT_create()
bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

garden = cv2.imread('Ghosts/lab7.png')
train = cv2.cvtColor(garden, cv2.COLOR_BGR2GRAY)
res = garden.copy()

def turn(image, angle):
    h, w = image.shape
    center = (w / 2, h / 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)               
    return cv2.warpAffine(image, rot_mat, (w, h), flags=cv2.INTER_LINEAR)

class Ghost:
    def __init__(self, image):
        self.image = image.copy()
        self.src_keypoints, self.src_descriptors = sift.detectAndCompute(self.image, None)
        self.dst_keypoints, self.dst_descriptors = sift.detectAndCompute(train, None)
        self.matches = sorted(bf.match(self.src_descriptors, self.dst_descriptors), key= lambda x: x.distance)

    def drawMatching(self):
        return cv2.drawMatches(self.image, self.src_keypoints, garden, self.dst_keypoints, self.matches[:50], garden, flags=2)
    
    def findIn(self, out_image):
        src_points = np.float32([self.src_keypoints[m.queryIdx].pt for m in self.matches]).reshape(-1, 1, 2)
        dst_points = np.float32([self.dst_keypoints[m.trainIdx].pt for m in self.matches]).reshape(-1, 1, 2)

        M, _ = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)

        h, w = self.image.shape
        pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)
    
        cv2.polylines(out_image, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
        cv2.fillPoly(train, [np.int32(dst)], 255, cv2.LINE_AA)

#candy_ghost
candy_ghost_img = cv2.imread('Ghosts/candy_ghost.png', 0)

candy_ghost = Ghost(candy_ghost_img)
candy_ghost.findIn(res)

candy_ghost_fliped = Ghost(cv2.flip(cv2.flip(candy_ghost_img, 0), 1))
candy_ghost_fliped.findIn(res)

candy_ghost_turned = Ghost(turn(candy_ghost_img, 15))
candy_ghost_turned.findIn(res)

#pampkin_ghost
pampkin_ghost = Ghost(cv2.imread('Ghosts/pampkin_ghost.png', 0))
pampkin_ghost.findIn(res)

#scary_ghost
scary_ghost_img = cv2.imread('Ghosts/scary_ghost.png', 0)

scary_ghost = Ghost(scary_ghost_img)
scary_ghost.findIn(res)

scary_ghost_fliped = Ghost(cv2.flip(scary_ghost_img, 1))
scary_ghost_fliped.findIn(res)

while True:
    cv2.imshow('Result', res)
    key = cv2.waitKey(0)
    if key == 27:
        break