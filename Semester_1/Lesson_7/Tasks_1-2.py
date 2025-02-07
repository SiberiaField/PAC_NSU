import numpy as np
import cv2
import os

def getPairs():
    pairs = []
    for _, _, fnames in os.walk(os.path.abspath("nails_segmentation\\images")):
        for name in fnames:
            img = cv2.imread("nails_segmentation\\images\\" + name)
            lbl = cv2.imread("nails_segmentation\\labels\\" + name)
            pairs.append([img, lbl])
    return pairs

def augmentation(pair, n, rng):
    text = ''
    pair = pair.copy()
    if n == 0: #Turn
        h, w = pair[0].shape[:2]
        center = (w / 2, h / 2)
        angle = rng.integers(0, 360)
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)               
        pair[0] = cv2.warpAffine(pair[0], rot_mat, (w, h), flags=cv2.INTER_LINEAR)
        pair[1] = cv2.warpAffine(pair[1], rot_mat, (w, h), flags=cv2.INTER_LINEAR)
        text = 'Turn ' + str(angle)
    elif n == 1: #Flip
        flipCode = rng.integers(0, 2)
        pair[0] = cv2.flip(pair[0], flipCode)
        pair[1] = cv2.flip(pair[1], flipCode)
        text = 'Flip'
        text += ' vert' if flipCode else ' hor'
    elif n == 2: #Crop
        bbox = rng.integers(10, 100, size=4)
        pair[0] = pair[0][bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
        pair[1] = pair[1][bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
        text = 'Crop'
    else: #Blur
        pair[0] = cv2.medianBlur(pair[0], 3)
        pair[1] = cv2.medianBlur(pair[1], 3)
        text = 'Blur'

    pair[0] = cv2.resize(pair[0], (256, 256))
    pair[1] = cv2.resize(pair[1], (256, 256))
    pair[0] = cv2.putText(pair[0], text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    return pair

def augmentPairs(pairs, n):
    res = []
    rng = np.random.default_rng()
    pair_indices = np.arange(len(pairs))
    res_indices = rng.choice(pair_indices, n, False)
    for i in res_indices:
        res.append(augmentation(pairs[i], rng.integers(4), rng))
    yield res

def main(n):
    pairs = getPairs()
    gen = augmentPairs(pairs, n)
    while True:
        for res in gen:
            for pair in res:
                cv2.imshow("image", pair[0])
                cv2.imshow("label", pair[1])
                key = cv2.waitKey(0)
                if key == 27:
                    break
        break

main(20)