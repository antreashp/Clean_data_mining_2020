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
import numpy as np

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
    average = []
    ticks = np.sort(list(set(processed_df[group])))
    for i in ticks:
        data.append(list(df[df[group]==i][var]))
        average.append(pd.mean(list(df[df[group]==i][var])))
    return average    
    length = []
    for i in range(0, len(data)):
        length.append(len(data[0]))
    if length[0] == max(length):
        print('balanced')
        print(length)
    else: print('unbalanced')
    fig, ax = plt.subplots(figsize=(9,4))
    ax.violinplot(data, showmeans=True)
    plt.xticks(np.arange(1, len(ticks)+1), ticks, rotation='vertical')
    #plt.title('distribution of '+var+' by '+group+'')
    ax.set_ylabel(''+var+'')
    plt.grid(axis='y')
    plt.show
 
average = violin(df=df)

violin(var='circumplex.arousal', group='user_id')  

#still figuring out how to remove the nans in the raw data
violin(df=df, var='mood', group='user_id') 

sns.distplot(df[df['user_id']=='AS14.01'].mood)

sns.distplot(processed_df['circumplex.arousal'])



sns.jointplot("mood", "activity", data=processed_df, kind="hex")    
processed_df[group]
