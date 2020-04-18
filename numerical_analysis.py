from preprosses import *
import numpy as np
import pandas as pd


preprocess_instance = preprocess(filename)
df = preprocess_instance.create_dataframe()

# print(df['mood'].mean() + 1)
# print(df.groupby('user_id')['mood'].mean() + 1)
print(df.groupby(['date', 'user_id'])['mood'].mean().sort_values().max() + 1)
print(df.groupby(['date', 'user_id'])['mood'].mean().sort_values().min() + 1)
