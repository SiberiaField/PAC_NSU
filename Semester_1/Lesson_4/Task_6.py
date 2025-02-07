import numpy as np
A = np.matrix("1 0 1; 0 1 0; 1 0 1")
U, S, Vh = np.linalg.svd(A)
print(f"\nLeft singular vectors:\n{U.transpose()}")
print(f"\nSingular values:\n{S}")
print(f"\nRight singular vectors:\n{Vh}")