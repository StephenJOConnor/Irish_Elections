#!/usr/bin/env python
# coding: utf-8

# In[52]:
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
import ast


# In[4]:
# load in Dail data and remove non-utf-8 symbols
xls = pd.read_csv('Dail.csv', encoding='cp1252')

for c in ['constit', 'party', 'name']:
    xls[c] = xls[c].apply(lambda x: str(x.encode('utf-8')). ##replace long dash
      replace('\\xe2\\x80\\x93', '-'). \
      replace('\\xc3\\xa1', 'á'). \
      replace('\\xc3\\xa9', 'é'). \
      replace('\\xc3\\xad','í'). \
      replace('\\xe2\\x80\\x99', "'"). \
      ##replace superflous b, spacing, and quotations
      replace("b'", ""). \
      replace('b"', ""). \
      replace('"', ""). \
      replace('\\xc3\\x9', 'Ó'). \
      replace('\\xc3\\xba', 'ú'). \
      replace("'","").rstrip())
# In[6]:

#encode unique parties, candidates, constituencies, races, and election
lb_make = LabelEncoder()
xls["party_code"] = lb_make.fit_transform(xls["party"])
xls["name_code"] = lb_make.fit_transform(xls["name"])
xls["constit_code"] = lb_make.fit_transform(xls["constit"])

## race is an unique race in a year/constituency/and election type
## elec_num is the national-level election signifier
xls['race'] = lb_make.fit_transform(xls.apply(lambda x: str(x['year']) \
   + str(x['constit_code']) + str(x['election']), axis = 1))
xls['elec_num'] = lb_make.fit_transform(xls.apply(lambda x: str(x['year']) \
   +  str(x['election']), axis = 1))

# In[8]:
# find the elmination rnd for each candidate
xls['elimination_rd'] = 0

# replace the transfers for the ceann comhairle (speaker of the house)
xls['transfers'] = xls['transfers'].apply(lambda x: ast.literal_eval(x.replace('CC', '[0]')))

# replace the round_totals for the CC
xls['round_totals'] = xls['round_totals'].apply(lambda x:  ast.literal_eval(x.replace('CC', '[0]')))

# In[10]:

def find_last_rnd(x):
    
    y = np.where(np.array(x['transfers'])<0)[0]
    z = np.where(np.array(x['round_totals'])==0)[0]
    # where surplus votes are exported after candidate is elected
    if len(y)>0:
        return int(y[0])
    # where candidate drops out
    elif len(z)>0:
        return int(z[0])
    # where candidate makes it to the final round without transferring votes out
    else:
        return 0

# find the round where the cadidate was eliminated (round is zero indexed)
xls['elimination_rd'] = xls.apply(lambda x: find_last_rnd(x), axis = 1)

# make type integer
xls['elimination_rd'] = xls['elimination_rd'].astype(int)

# In[11]:

# finds the votes transferred from the candidate 
xls['vote_out_1'] = xls.apply(lambda x: sum(n for n in [x['transfers'][x['elimination_rd']]] if x['elimination_rd']>0), axis=1)
xls['vote_out_2'] = xls.apply(lambda x: sum(n for n in [x['round_totals'][x['elimination_rd']-1]] if (x['round_totals'][-1]==0)), axis=1)
xls['vote_out'] = xls.apply(lambda x: abs(x['vote_out_1']) if x['vote_out_1']!=0 else x['vote_out_2'], axis=1)

# drop vote_out 1 and 2
xls = xls.drop(['vote_out_1', 'vote_out_2'], axis=1)

# In[]
# pickle xls
xls.to_pickle("./dail_election_data.pkl")