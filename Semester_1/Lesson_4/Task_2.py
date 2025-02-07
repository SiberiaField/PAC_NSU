import numpy as np
h = int(input())
w = int(input())
picture = np.random.randint(0, 255, size=(h, w))
print(np.unique(picture).size)