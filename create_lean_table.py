import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
import ast
import matplotlib.pyplot as plt


# In[4]:

dail = pd.read_pickle('./dail_election_data.pkl')

dail['rounds'] = dail['transfers'].apply(lambda x: len(x))

# In[]

#create function to find inward vote within list x for index i
def inward_votes(x, i):
    if len(x) > i and len(x)>0:
        if x[i] > 0:
            return x[i]
        else:
            return 0
    else:
        return 0
# In[]

# create columns for each inward vote-transfer
for i in range(0, max(dail['rounds'])):   
    dail['inward_vote_' + str(i)] = dail.apply(lambda x: inward_votes(x['transfers'], i), axis = 1)

# In[16]:

# 
for i in range(1, max(dail['rounds'])):
	
	# find the parties eliminated at each round
    rnd_transfers = dail[dail['elimination_rd']==i][['race', 'party']]    
    rnd_transfers = rnd_transfers.drop_duplicates()
    rnd_transfers['party_transfer_' + str(i)] = rnd_transfers.\
        apply(lambda x: sorted(list(rnd_transfers[rnd_transfers['race']== x['race']]['party'])), axis=1)
    rnd_transfers = rnd_transfers.drop(['party'], axis=1)
	
	# find how many votes were transfered to each candidate by eliminated party
    rnd_transfers['party_transfer_' + str(i)] = rnd_transfers['party_transfer_' + str(i)].\
        apply(lambda x: str(x) if x is not None else "[]")
        
    rnd_transfers = rnd_transfers.drop_duplicates()
        
    rnd_transfers['party_transfer_' + str(i)] = rnd_transfers['party_transfer_' + str(i)].astype('str')
    
    dail = pd.merge(dail, rnd_transfers, how = 'left', on = 'race')

# In[20]:

# reference table for party names and codes
party_codes = dail[['party', 'party_code']].drop_duplicates().sort_values(by='party').reset_index(drop=True)

# In[]

for i in range(1, max(dail['rounds'])):
    dail['party_transfer_' + str(i)] = dail['party_transfer_' + str(i)].apply(lambda x: '[]' if str(x)=='nan' else x)

# In[]

# isolate single party transfers
def individual_party(x):
    x = ast.literal_eval(x)
    
    if isinstance(x, list):
        if len(x)==1:
            return int(party_codes[party_codes['party']==x[0]]['party_code'])
    else:
        return None
	
# In[]
for i in range(1, max(dail['rounds'])):
    
    # find the single party who transfered votes
    dail['single_party_transfer_' + str(i)] = dail['party_transfer_' + str(i)].apply(lambda x: individual_party(x))
    
    # total vote from that or all transferring parties
    vote_out_ag = dail[dail['elimination_rd']==i][['race', 'vote_out']]
    vote_out_ag = pd.DataFrame(vote_out_ag.groupby(['race'], as_index = False).sum())
    vote_out_ag['agg_vote_out_' + str(i)] = vote_out_ag['vote_out']
    vote_out_ag = vote_out_ag.drop('vote_out', axis = 1)
    dail = pd.merge(dail, vote_out_ag, how = 'left', on = 'race')
    dail['agg_vote_out_' + str(i)] = dail['agg_vote_out_' + str(i)].fillna(0).astype(int)

# In[25]:

first_pref_trans = pd.DataFrame(dail.groupby(['elec_num', 'party_code', 'single_party_transfer_1'], \
    as_index=False)['inward_vote_1', 'agg_vote_out_1'].sum())

first_pref_trans['pct_trans'] = first_pref_trans['inward_vote_1']/first_pref_trans['agg_vote_out_1']

# In[27]:

# find the total first prefs per election
total = dail[['elec_num', 'race', 'valid']].drop_duplicates()
total = pd.DataFrame(total.groupby(['elec_num'], \
    as_index=False)['valid'].sum())

# find party share of first preferences
first_pref_count = pd.DataFrame(dail.groupby(['elec_num', 'party_code'], \
    as_index=False)['inward_vote_0'].sum())
first_pref_count = pd.merge(first_pref_count, total, on = 'elec_num', how = 'left')
first_pref_count['first_pref_pct'] = first_pref_count['inward_vote_0']/first_pref_count['valid']

first_pref_count = pd.merge(first_pref_count, first_pref_trans[first_pref_trans['party_code']==first_pref_trans['single_party_transfer_1']][['party_code', 'elec_num', 'pct_trans']]\
                            , how = "left", on = ['elec_num','party_code'])


# In[27]:
first_pref_trans['num_parties']=first_pref_trans['party_transfer_1'].apply(lambda x:                                                        len(x))


# In[28]:


first_pref_trans = first_pref_trans[(first_pref_trans['num_parties']==1) & (first_pref_trans['pct_trans']<1)]


# In[29]:


first_pref_trans['party_transfer_1'] = first_pref_trans['party_transfer_1'].apply(lambda x: x[0])


# In[30]:


fg_lp_collapsed = dail
fg_lp_collapsed['party_transfer_1'] = fg_lp_collapsed['party_transfer_1'].apply(lambda x: 'FG/LP' if (x=="['Fine Gael']")                                                                                | (x=="['Labour Party']") else x)


# In[31]:


fg_lp_collapsed = pd.DataFrame(fg_lp_collapsed.groupby(['elec_num', 'party', 'party_transfer_1'],                                             as_index=False)['inward_vote_1', 'agg_vote_out_1'].sum())
fg_lp_collapsed['pct_trans'] = fg_lp_collapsed['inward_vote_1']/fg_lp_collapsed['agg_vote_out_1']
fg_lp_collapsed = fg_lp_collapsed[(fg_lp_collapsed['pct_trans']<1) & (fg_lp_collapsed['party_transfer_1']=='FG/LP')]


# In[86]:


healy_rae = dail[(dail['constit']=='Kerry')]
healy_rae['party'] = healy_rae.apply(lambda x: 'Healy-Rae' if (x['name_code']==1426)|(x['name_code']==374) else x['party'], axis=1)


# In[87]:


healy_rae = healy_rae.drop(['election', 'year', 'constit', 'party_code', 'name_code', 'constit_code',                             'race', 'elec_num', 'party_transfer_1', 'agg_vote_out_1', 'p_t_1_code'], axis=1)


# In[106]:


kerry_trans_1 = healy_rae[healy_rae['elimination_rd']==1][['party']] 
kerry_trans_1 = kerry_trans_1.drop_duplicates()
healy_rae['party_transfer_1']='Healy-Rae'


# In[109]:


ag_v_out=healy_rae.groupby(['elimination_rd'], as_index=False)['vote_out'].sum()
healy_rae['agg_vote_out_1']=ag_v_out[]


# In[32]:


##Fianna Fail's inward transfers
ff_ff = first_pref_trans[(first_pref_trans['party']=='Fianna Fáil') &                        (first_pref_trans['party_transfer_1']=='Fianna Fáil')]
ff_fg = first_pref_trans[(first_pref_trans['party']=='Fianna Fáil') &                        (first_pref_trans['party_transfer_1']=='Fine Gael')]
ff_lp = first_pref_trans[(first_pref_trans['party']=='Fianna Fáil') &                        (first_pref_trans['party_transfer_1']=='Labour Party')]
ff_ind = first_pref_trans[(first_pref_trans['party']=='Fianna Fáil') &                        (first_pref_trans['party_transfer_1']=='Independent')]
ff_sf = first_pref_trans[(first_pref_trans['party']=='Fianna Fáil') &                        (first_pref_trans['party_transfer_1']=='Sinn Féin')]

##By top three parties plus independents
ff_ff_x, ff_ff_y = ff_ff['elec_num'], ff_ff['pct_trans']
ff_fg_x, ff_fg_y = ff_fg['elec_num'], ff_fg['pct_trans']
ff_lp_x, ff_lp_y = ff_lp['elec_num'], ff_lp['pct_trans']
ff_ind_x, ff_ind_y = ff_ind['elec_num'], ff_ind['pct_trans']
ff_sf_x, ff_sf_y = ff_sf['elec_num'], ff_sf['pct_trans']

##plotting
plt.plot(ff_ff_x, ff_ff_y, color='g')
plt.plot(ff_fg_x, ff_fg_y, color='blue')
plt.plot(ff_lp_x, ff_lp_y, color='red')
plt.plot(ff_ind_x, ff_ind_y, color='black')
plt.plot(ff_sf_x, ff_sf_y, color='yellow')
plt.xlabel('Elections')
plt.ylabel('Transfers')
plt.title('1st Pref Transfers to FF')
plt.show


# In[33]:


##Labour's inward transfers
lp_ff = first_pref_trans[(first_pref_trans['party']=='Labour Party') &                        (first_pref_trans['party_transfer_1']=='Fianna Fáil')]
lp_fg = first_pref_trans[(first_pref_trans['party']=='Labour Party') &                        (first_pref_trans['party_transfer_1']=='Fine Gael')]
lp_lp = first_pref_trans[(first_pref_trans['party']=='Labour Party') &                        (first_pref_trans['party_transfer_1']=='Labour Party')]
lp_ind = first_pref_trans[(first_pref_trans['party']=='Labour Party') &                        (first_pref_trans['party_transfer_1']=='Independent')]

##By top three parties plus independents
lp_ff_x, lp_ff_y = lp_ff['elec_num'], lp_ff['pct_trans']
lp_fg_x, lp_fg_y = lp_fg['elec_num'], lp_fg['pct_trans']
lp_lp_x, lp_lp_y = lp_lp['elec_num'], lp_lp['pct_trans']
lp_ind_x, lp_ind_y = lp_ind['elec_num'], lp_ind['pct_trans']

##plotting
plt.plot(lp_ff_x, lp_ff_y, color='g')
plt.plot(lp_fg_x, lp_fg_y, color='blue')
plt.plot(lp_lp_x, lp_lp_y, color='red')
plt.plot(lp_ind_x, lp_ind_y, color='black')
plt.xlabel('Elections')
plt.ylabel('Transfers')
plt.title('1st Pref Transfers to LP')
plt.show


# In[34]:


##Independents' inward transfers
ind_ff = first_pref_trans[(first_pref_trans['party']=='Independent') &                        (first_pref_trans['party_transfer_1']=='Fianna Fáil')]
ind_fg = first_pref_trans[(first_pref_trans['party']=='Independent') &                        (first_pref_trans['party_transfer_1']=='Fine Gael')]
ind_lp = first_pref_trans[(first_pref_trans['party']=='Independent') &                        (first_pref_trans['party_transfer_1']=='Labour Party')]
ind_ind = first_pref_trans[(first_pref_trans['party']=='Independent') &                        (first_pref_trans['party_transfer_1']=='Independent')]

##By top three parties plus independents
ind_ff_x, ind_ff_y = ind_ff['elec_num'], ind_ff['pct_trans']
ind_fg_x, ind_fg_y = ind_fg['elec_num'], ind_fg['pct_trans']
ind_lp_x, ind_lp_y = ind_lp['elec_num'], ind_lp['pct_trans']
ind_ind_x, ind_ind_y = ind_ind['elec_num'], ind_ind['pct_trans']

##plotting
plt.plot(ind_ff_x, ind_ff_y, color='g')
plt.plot(ind_fg_x, ind_fg_y, color='blue')
plt.plot(ind_lp_x, ind_lp_y, color='red')
plt.plot(ind_ind_x, ind_ind_y, color='black')
plt.xlabel('Elections')
plt.ylabel('Transfers')
plt.title('1st Pref Transfers to Independents')
plt.show


# In[38]:


##Fine Gael's inward transfers
fg_ff = first_pref_trans[(first_pref_trans['party']=='Fine Gael') &                        (first_pref_trans['party_transfer_1']=='Fianna Fáil')]
fg_fg = first_pref_trans[(first_pref_trans['party']=='Fine Gael') &                        (first_pref_trans['party_transfer_1']=='Fine Gael')]
fg_lp = first_pref_trans[(first_pref_trans['party']=='Fine Gael') &                        (first_pref_trans['party_transfer_1']=='Labour Party')]
fg_ind = first_pref_trans[(first_pref_trans['party']=='Fine Gael') &                        (first_pref_trans['party_transfer_1']=='Independent')]
fg_fglp = fg_lp_collapsed[(fg_lp_collapsed['party']=='Fine Gael') &                        (fg_lp_collapsed['party_transfer_1']=='FG/LP')]

##By top three parties plus independents
fg_ff_x, fg_ff_y = fg_ff['elec_num'], fg_ff['pct_trans']
fg_fg_x, fg_fg_y = fg_fg['elec_num'], fg_fg['pct_trans']
fg_lp_x, fg_lp_y = fg_lp['elec_num'], fg_lp['pct_trans']
fg_ind_x, fg_ind_y = fg_ind['elec_num'], fg_ind['pct_trans']
fg_fglp_x, fg_fglp_y = fg_fglp['elec_num'], fg_fglp['pct_trans']

##plotting
plt.plot(fg_ff_x, fg_ff_y, color='g')
#plt.plot(fg_fg_x, fg_fg_y, color='blue')
#plt.plot(fg_lp_x, fg_lp_y, color='red')
plt.plot(fg_ind_x, fg_ind_y, color='black')
plt.plot(fg_fglp_x, fg_fglp_y, color='purple')

plt.xlabel('Elections')
plt.ylabel('Transfers')
plt.title('1st Pref Transfers to FG')
plt.show


# In[39]:


##SF's inward vote
sf_ff = first_pref_trans[(first_pref_trans['party']=='Sinn Féin') &                        (first_pref_trans['party_transfer_1']=='Fianna Fáil')]
sf_fg = first_pref_trans[(first_pref_trans['party']=='Sinn Féin') &                        (first_pref_trans['party_transfer_1']=='Fine Gael')]
sf_lp = first_pref_trans[(first_pref_trans['party']=='Sinn Féin') &                        (first_pref_trans['party_transfer_1']=='Labour Party')]
sf_ind = first_pref_trans[(first_pref_trans['party']=='Sinn Féin') &                        (first_pref_trans['party_transfer_1']=='Independent')]
sf_fglp = fg_lp_collapsed[(fg_lp_collapsed['party']=='Sinn Féin') &                        (fg_lp_collapsed['party_transfer_1']=='FG/LP')]

##By top three parties plus independents
sf_ff_x, sf_ff_y = sf_ff['elec_num'], sf_ff['pct_trans']
sf_fg_x, sf_fg_y = sf_fg['elec_num'], sf_fg['pct_trans']
sf_lp_x, sf_lp_y = sf_lp['elec_num'], sf_lp['pct_trans']
sf_ind_x, sf_ind_y = sf_ind['elec_num'], sf_ind['pct_trans']
sf_fglp_x, sf_fglp_y = sf_fglp['elec_num'], sf_fglp['pct_trans']


# In[51]:


fit = np.polyfit(fg_fglp_x, fg_fglp_y, 1)
fit_fn = np.poly1d(fit)
plt.plot(fg_fglp_x, fg_fglp_y, 'yo', fg_fglp_x, fit_fn(fg_fglp_x), '--k', color = 'purple')

fit = np.polyfit(fg_ff_x, fg_ff_y, 1)
fit_fn = np.poly1d(fit)
plt.plot(fg_ff_x, fg_ff_y, 'yo', fg_ff_x, fit_fn(fg_ff_x), '--k', color = 'g')

fit = np.polyfit(ff_ff_x, ff_ff_y, 1)
fit_fn = np.poly1d(fit)
plt.plot(ff_ff_x, ff_ff_y, 'yo', ff_ff_x, fit_fn(ff_ff_x), '--k', color='b')

#fit = np.polyfit(ff_ind_x, ff_ind_y, 1)
#fit_fn = np.poly1d(fit)
#plt.plot(ff_ind_x, ff_ind_y, 'yo', ff_ind_x, fit_fn(ff_ind_x), '--k', color='indigo')

#fit = np.polyfit(ind_ff_x, ind_ff_y, 1)
#fit_fn = np.poly1d(fit)
#plt.plot(ind_ff_x, ind_ff_y, 'yo', ind_ff_x, fit_fn(ind_ff_x), '--k', color='cyan')

#fit = np.polyfit(ff_fg_x, ff_fg_y, 1)
#fit_fn = np.poly1d(fit)
#plt.plot(ff_fg_x, ff_fg_y, 'yo', ff_fg_x, fit_fn(ff_fg_x), '--k', color='red')

fit = np.polyfit(sf_ff_x, sf_ff_y, 1)
fit_fn = np.poly1d(fit)
plt.plot(sf_ff_x, sf_ff_y, 'yo', sf_ff_x, fit_fn(sf_ff_x), '--k', color='magenta')

fit = np.polyfit(ff_sf_x, ff_sf_y, 1)
fit_fn = np.poly1d(fit)
plt.plot(ff_sf_x, ff_sf_y, 'yo', ff_sf_x, fit_fn(ff_sf_x), '--k', color='black')

fit = np.polyfit(ff_lp_x, ff_lp_y, 1)
fit_fn = np.poly1d(fit)
plt.plot(ff_lp_x, ff_lp_y, 'yo', ff_lp_x, fit_fn(ff_lp_x), '--k', color='orange')

plt.show


# In[256]:


dail.groupby(['elec_num', 'year'], as_index=False).size()


# In[259]:


first_pref_trans.groupby(['party'], as_index=False).size()


# In[94]:


dail


# In[ ]:




