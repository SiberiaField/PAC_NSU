import numpy as np

def solve(matrix, equations):
    matrix = np.linalg.inv(matrix)
    return np.dot(matrix, equations)

matrix = np.array( [[3, 4, 2],
                    [5, 2, 3],
                    [4, 3, 2]])

equations = np.array([17, 23, 19])

print(solve(matrix, equations))