import pandas as pd
import numpy as np

df = pd.read_csv("titanic_with_labels.csv", sep=' ')

#Point 1
df = df.replace({'sex': {r'^[нН-]': np.nan}}, regex=True)
df.dropna(inplace=True, subset='sex')
df = df.replace({'sex': {r'^[жЖ]': 0, r'^[мMМ]': 1}}, regex=True)

#Point 2
df.fillna({'row_number': df['row_number'].max()}, inplace=True)

#Point 3
third_quantile = df['liters_drunk'].quantile(0.75)
second_quantile = df['liters_drunk'].quantile(0.5)
print(f'third_quantile: {third_quantile}')
print(f'second_quantile: {second_quantile}\n')

cond = lambda value: (value >= 0) and (value <= third_quantile)
mask = df['liters_drunk'].apply(cond)
df['liters_drunk'] = df['liters_drunk'].where(mask, second_quantile)
print(df)