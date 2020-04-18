# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 20:16:16 2020

@author: Robert
"""

import pandas as pd
from preprosses import  preprocess
import datetime
import matplotlib as plt
import seaborn as sns

filename = 'data/RAW_Data.pickle'
methods = ['average','max','max','max','max','max','max','max','max','max',
'max','max','max','max','max','max','max','max','average','average',
'average','average','average','average','average','average','average','average']
preprocess_instance = preprocess(filename, window_size=1, methods=methods)
preprocess_instance.normalize()
preprocess_instance.bin()
df = preprocess_instance.create_dataframe()
processed_df = preprocess_instance.create_dataframe_pros()
print(df.head(1))
print(processed_df.head(1))
print(processed_df.columns)

def violin(df=processed_df, var='mood', group='user_id'):
    data = []
    ticks = []
    for i in set(processed_df[group]):
        ticks.append(i)
        l = list(df[df[group]==i][var])
        data.append(l)
    length = []
    for i in range(0, len(data)):
        length.append(len(data[0]))
    if length[0] == max(length):
        print('balanced')
    else: print('unbalanced')
    fig, ax = plt.subplots(figsize=(9,4))
    ax.violinplot(data, showmeans=True)
    plt.xticks(np.arange(1, len(ticks)+1), ticks, rotation='vertical')
    plt.title('distribution of '+var+' by '+group+'')
    ax.set_ylabel(''+var+'')
    plt.grid(axis='y')
    plt.show

violin()    


sns.distplot(processed_df.date)

sns.distplot(date)

    