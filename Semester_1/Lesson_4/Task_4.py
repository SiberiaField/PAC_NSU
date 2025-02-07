import numpy as np

def isTriangle(line):
    if line[0] < line[1] + line[2] and \
    line[1] < line[0] + line[2] and \
    line[2] < line[0] + line[1]:
        return True
    else: 
        return False

n = 6
matrix = np.random.randint(1, 20, size=(n, 3))
print(matrix, '\n')

mask = np.apply_along_axis(isTriangle, 1, matrix)
print(matrix[mask])