import numpy as np
import os,sys
from collections import defaultdict
from tqdm import tqdm
# from datetime import datetime
from datetime import date as date_lib
import datetime
import pandas as pd

filename = 'C:/Users/Robert/Documents/VU - economics/DataMiningTechniques/data/dataset_mood_smartphone.csv'
filenameDEFAULT = '../dataset_mood_smartphone.csv'
with open (filename,'r') as f: 
    data = f.readlines()

data = [data[i].replace("'",'').replace('"','').replace('\n','').split(',') for i in range(len(data))]
    
data_header = data[0]
data = data[1:]
print(data_header)
print(data[:10])

"""NOTES ROBERT """
data_rs = np.reshape(data, (len(data),5))
data_rs = pd.DataFrame(data_rs)
data_rs.columns = data_header
data_rs['stamp'] = data_rs['time']
data_rs['date'] = data_rs.time.str.split(' ', expand=True)[0]
data_rs['time'] = data_rs.time.str.split(' ', expand=True)[1]
"""END"""
 
data_table = [[],[],[],[],[]]

for i in range(len(data)):
    data_table[0].append(data[i][0]) 
    data_table[1].append(data[i][1]) 
    data_table[2].append(data[i][2]) 
    data_table[3].append(data[i][3]) 
    data_table[4].append(data[i][4])    

# print(data_table[1][:50])
data_table = np.array(data_table)
# meh = data_table[data_table=='AS14.01']
# meh = [data_table[1][i] if data_table[1][i] == 'AS14.01' else continue  for i in range(len(data_table[1])) ] 
# print(len(meh))

# print(list(set(list(data_table[1]))))

# print(len(list(set(list(data_table[1])))))
# for i in range(10):
#     meh = data_table[data_table=='AS14.0'+str(i)]
#     # meh = [data_table[1][i] if data_table[1][i] == 'AS14.01' else continue  for i in range(len(data_table[1])) ] 
#     print(len(meh))

def decode_date(d):
    """
    Decode date into day, season, part of the day
    :param d: str
    :return: (int days, int season, int part_of_the_day)
      days: number of days since 1-1-1
      season: 0 winter, 1 spring, 2 summer, 3 autumn
      part_of_the_day: 0 midnight, 1 morning, 2 afternoon, 3 evening
    """
    date,time = d.split(' ')

    #print(date,time)
    year,month,day = date.split('-')
    year,month,day = int(year),int(month),int(day)
    hours,minutes,seconds = time.split('.')[0].split(':')
    
    hours,minutes,seconds  = int(hours),int(minutes),int(seconds)
    # sub = datetime.fromisoformat('2014-03-05 07:22:26.976637+00:00').timestamp() -datetime.fromisoformat('2014-03-06 07:22:26.976637+00:00').timestamp()
    # print(int(''.join(c for c in '2014-03-04 07:22:26.976637+00:00' if c.isdigit())) - int(''.join(c for c in '2014-03-06 07:22:26.976637+00:00' if c.isdigit())))
    # print(sub)
    
    # 100000*year + 10000*month + 1000 *day + 100 
    # print((date_lib(year,month,day) - date_lib(1,1,1)).days)
    # print((a-b).days)
    date = date_lib(year,month,day)

    part_of_the_day = None
    if hours <6: 
        part_of_the_day = (1,0,0,0)
    elif hours <12:
        part_of_the_day = (0,1,0,0)
    elif hours <18:
        part_of_the_day = (0,0,1,0)
    else:
        part_of_the_day = (0,0,0,1)

    season = None
    # if month == 2 and day > 10:
    #     season = (day/31.,0,0,0,0,0)
    # if month == 3 and day < 15:
    #     season = (0,day/31.,0,0,0,0)
    # elif month == 3 and day > 15:
    #     season = (0,0,day/31.,0,0,0)
    # elif month == 4 and day < 15:
    #     season = (0,0,day/31.,0,0,0)
    # elif month == 4 and day > 15:
    #     season = (0,0,0,day/31.,0,0)    
    # elif month == 5 and day < 15:
    #     season = (0,0,0,day/31.,0,0)
    # elif month == 5 and day > 15:
    #     season = (0,0,0,0,day/31.,0)
    # elif month == 6 and day < 15:
    #     season = (0,0,0,0,day/31.,0)

    season = [0,0,0,0,0]
    if day >15:
            season[month-2 ]= day/31.
            season[month-2 +1]=1 - ( day/31.)
        
    else:
        season[month-2 ]= day/31.
        season[month-2 -1]=1 - ( day/31.)
    
    
    # if month <= 2  or  month == 12: 
    #     season = (1,0,0,0)
    # elif month < 6:
    #     season = (0,1,0,0)
    # elif month < 9:
    #     season = (0,0,1,0)
    # else:
    #     season = (0,0,0,1)
        
    return (date_lib(year,month,day) - date_lib(1,1,1)).days , part_of_the_day , season   "see question 3___ below, why use data_lab(1,1,1) as starting point an not first record

print('lowest dummy', min(data_rs.stamp), decode_date(min(data_rs.stamp))[2])
"1____why does this make sense as lowest value? why not [1,0,0,0,0]

print('higest dummy', max(data_rs.stamp), decode_date(max(data_rs.stamp))[2])
"2___likewise why does this make sense as highest value? why not [0,0,0,0,1]

print(datetime.datetime.strptime(max(data_rs.date), '%Y-%m-%d')-datetime.datetime.strptime(min(data_rs.date), '%Y-%m-%d'))
"3___the total timespan is 112 days - why can't we just use a single float ranging from 1/112 to 112/112 (since in the dataset were are only approaching better weather)
"4___other idea: we could match the dates with the Dutch temperature record or amount of sunlight

# exit()
"5___is this formula ever used?"
def group_datapoints_by_id (d_table): 

    
    ids = list(set(list(d_table[1])))
    data_dict = defaultdict(list)
    for id in ids:

        # print(id)
        # print(np.where(d_table==id))
        # print(d_table[:,np.where(d_table==id)[1]])
        all_records_of_id = d_table[:,np.where(d_table==id)[1]]
        data_dict[id] = all_records_of_id
    return data_dict


def var_handler(var,val):
    var_ids =[ 'mood'  , 'circumplex.arousal','circumplex.valence' ,'activity','screen' ,'call','sms' ,'appCat.builtin' ,'appCat.communication', 'appCat.entertainment','appCat.finance' ,'appCat.game','appCat.office','appCat.other','appCat.social' ,'appCat.travel' ,'appCat.unknown' ,'appCat.utilities', 'appCat.weather' ]
    idx = var_ids.index(var)
    data = [None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None]
    if val == 'NA':
        return data
    val = float(val)
    if idx ==0:
        data[idx] = val-1.
        return data
    elif idx ==1 or idx == 2 :
        data[idx] = (val+2)/4
        return data
    else:
        data[idx] = val
        return data
    # return 
    
""" NOTES ROBERT """
data_table.transpose(1,0)
data_table

for i , item in tqdm(enumerate(data_table.transpose(1,0)[0])):
    #print(i)
    print(item)
    i,id,d,var,val = item
    
a, b, c, d, e = ['1', 'AS14.01', '2014-02-26 13:00:00.000', 'mood', '6']
    
""" END """




"----steps---"
"1) seperate each item in array in 'i','id','d','var','val'
"2) var-handler: (I) replace NA with None; (II) for mood conver range[1,10] to [0,9] WHY?; (III) for arousal and valence range[-2,2] to [0,1]
"check scale of 'call' variable:
data_rs[['variable','value']][data_rs.variable=='call'].head(10) 
"3) decode date in 'date_idx'; 'part_of_day';  'season'
"4)
"___6 what is this defaultdict business?
   
def group_datapoints_by_day_and_user (d_table):
 
    # dates = list(set(list(d_table[2])))
    # ids = list(set(list(d_table[1])))
    data_dict = defaultdict(None)
    for i , item in tqdm(enumerate(d_table.transpose(1,0))):
        # print(item.shape)
        i,id,d,var,val = item
        # print(i,id,d,var,val)
        # exit()
        var_list = var_handler(var,val)
        
        date_idx,part_of_day,season = decode_date(d)
        mylist =  []
        if id not in data_dict.keys():
            data_dict[id] = defaultdict(list)
        # elif var not in data_dict[id].keys()
        
        for part in part_of_day:
            
            var_list.append(part)
        for s in season:
            
            var_list.append(s)
        
        
        # all_records_of_id = d_table[:,np.where(d_table==id)[1]]
        data_dict[id][date_idx].append(var_list)
    return data_dict
print(data_table.shape)

"5) make final dataset
d_dict_by_id = group_datapoints_by_day_and_user (data_table)

type(d_dict_by_id)

"6) save to pickle"
def save(meh):

    import pickle
    with open('C:/Users/Robert/Documents/VU - economics/DataMiningTechniques/data/RAW_Data.pickle', 'wb') as f:
        pickle.dump(meh, f)
save(d_dict_by_id)
# print(d_dict_by_id[735325])
with open('../text.txt','w') as f:
    for id in (d_dict_by_id):
        for i,date in enumerate(d_dict_by_id[id]):
            # print(str(id)+str(d_dict_by_id[date][id])+'\n')
            f.write(str(i)+'_'+str(id)+str(d_dict_by_id[id][date])+'\n')
            
            
for i , item in tqdm(enumerate(d_table.transpose(1,0))):
# --------------------------------------------
# 1. 1 day per datapoint

#             user   season 
# day 
# --------------------------------------------------

# 2. splitting the day per datapoint

#             user   season 
# morning
# noon
# afternoon

# 3. splitting the day for each user per datapoint

#            season 
# morning
# noon
# afternoon



