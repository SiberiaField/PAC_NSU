import numpy as np
import pandas as pd

def condition_mean(row):
    row = row.values
    return np.mean(row, where= row > 0.3)

rng = np.random.default_rng()
df = pd.DataFrame(rng.uniform(0, 1, (5, 10)))
print(f"Frame:\n{df}\n")
df = df.apply(condition_mean, 1)
print(f"Res:\n{df}\n")