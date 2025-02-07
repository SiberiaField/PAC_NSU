import numpy as np
window_size = 3
vec = np.random.randint(50, size=10)
print(vec)
moving_vec = np.repeat(1 / window_size, window_size)
print(np.convolve(vec, moving_vec, 'valid'))