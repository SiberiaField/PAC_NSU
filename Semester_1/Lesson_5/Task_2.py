import numpy as np
import pandas as pd

def count_months(row):
   diff = pd.to_datetime(row['CompletionDate']) - pd.to_datetime(row['FirstProductionDate'])
   return int(diff / np.timedelta64(30, 'D'))

df = pd.read_csv("wells_info.csv")
df = df.set_index('API')
res = pd.DataFrame(df.apply(count_months, 1), columns=['WorkingMonths'])
print(res)