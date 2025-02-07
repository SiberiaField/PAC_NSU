import pandas as pd
import numpy as np

def get_max_counts(collumn: pd.Series):
    return collumn.value_counts().index[0]

df = pd.read_csv("wells_info_na.csv")
df['CompletionDate'] = pd.to_datetime(df['CompletionDate'])
df['FirstProductionDate'] = pd.to_datetime(df['FirstProductionDate'])

meds = df.median(numeric_only=True)
max_counts = df.drop(meds.index, axis=1).apply(get_max_counts)
replaces = pd.concat([meds, max_counts])
print(df.fillna(replaces))