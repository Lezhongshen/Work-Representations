#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# pip install --upgrade pandas==1.3.4


# In[159]:


import pandas as pd
import re


# In[2]:


df = pd.read_pickle('sanction_data.pkl')


# In[3]:


# df[df['Twitter_handles'] == 'lidiafopereira']


# In[4]:


# find out the those names that don't match with the handles

a = df[['Twitter_handles','Name']].groupby(['Twitter_handles','Name'], as_index = False).sum()
a.loc[a['Twitter_handles'].duplicated()==True]


# In[ ]:


# a.loc[a['Twitter_handles'] == 'lidiafopereira'] 
# a.loc[a['Twitter_handles'] == 'emmawiesner']
# a.loc[a['Twitter_handles'] == 'JMFernandesEU']
# a.loc[a['Twitter_handles'] == 'GrisetCatherine']


# In[ ]:


# df.loc[df['Twitter_handles'] == 'lidiafopereira']['Country'].values


# In[ ]:





# In[279]:


# # fix lidiafopereira
# df.loc[df['Twitter_handles'] == 'lidiafopereira','EU Party'] = "Group of the European People's Party (Christian Democrats)"
# df.loc[df['Twitter_handles'] == 'lidiafopereira','Name'] = "lídia pereira"
# df.loc[df['Twitter_handles'] == 'lidiafopereira']


# In[ ]:





# In[280]:


# # fix emma wiesner
# df.loc[df['Twitter_handles'] == 'emmawiesner','Name'] = "emma wiesner"
# df.loc[df['Twitter_handles'] == 'emmawiesner','EU Party'] = "Renew Europe Group"
# df.loc[df['Twitter_handles'] == 'emmawiesner','Country'] = "Sweden"
# df.loc[df['Twitter_handles'] == 'emmawiesner']


# In[ ]:





# In[281]:


# # fix josé manuel fernandes

# df.loc[df['Twitter_handles'] == 'JMFernandesEU','EU Party'] = "Group of the European People's Party (Christian Democrats)"
# df.loc[df['Twitter_handles'] == 'JMFernandesEU','Name'] = "josé manuel fernandes"
# df.loc[df['Twitter_handles'] == 'JMFernandesEU','Country'] = "Portugal"
# df.loc[df['Twitter_handles'] == 'JMFernandesEU']


# In[ ]:





# In[282]:


# # fix catherine griset
# df.loc[df['Twitter_handles'] == 'GrisetCatherine','EU Party'] = "Identity and Democracy Group"
# df.loc[df['Twitter_handles'] == 'GrisetCatherine','Name'] = "catherine griset"
# df.loc[df['Twitter_handles'] == 'GrisetCatherine']



# In[173]:


df.to_pickle('sanction_data_update.pkl')



# In[160]:


df = pd.read_pickle('sanction_data_update.pkl')




# In[4]:


df


# In[5]:


# retweet = []
# for element in df['Tweet']:
#     if 'RT' in element:
#         retweet.append(1)
#     else:
#         retweet.append(0)
        

# df['is_retweet'] = retweet


# In[6]:


df_retweet = df.loc[df['is_retweet'] == 1].copy()
df_retweet


# In[7]:


people = []
for x in df_retweet['Tweet']:
    s = re.findall("@([a-zA-Z0-9]{1,15})", x)
    people.append(s)


# In[283]:


#people


# In[9]:


people_retweet = []
for x in people:
    if x == []:
        people_retweet.append('')
    else:
        people_retweet.append(x[0])
        


# In[284]:


#people_retweet


# In[11]:


df_retweet['retweet_mention'] = people_retweet


# In[285]:


#df_retweet


# In[13]:


twitter_handle = set(df_retweet['Twitter_handles'])


# In[14]:


df_retweet_eu = df_retweet.loc[df_retweet['retweet_mention'].isin(twitter_handle)]


# In[286]:


#df_retweet_eu


# In[161]:


name_df = df[['Name','Twitter_handles']]
grouped = name_df.groupby('Twitter_handles')['Name'].agg("first")
dict_name = grouped.to_dict()


# In[287]:


#dict_name



# In[16]:


retweet_network = df_retweet_eu[['Twitter_handles','retweet_mention','Country','EU Party']].copy()
retweet_network


# In[17]:


retweet_network['weight'] = 1


# In[288]:


#retweet_network


# In[19]:


retweet_network = retweet_network.loc[retweet_network['Twitter_handles'] != retweet_network['retweet_mention']]


# In[289]:


#retweet_network


# In[166]:


retweet_network2 = retweet_network.groupby(['Twitter_handles','retweet_mention','Country','EU Party'], as_index = False).sum()


# In[290]:


#retweet_network2


# In[168]:


retweet_network2['Twitter_handles'] = [dict_name[x] for x in retweet_network2['Twitter_handles']]


# In[169]:


retweet_network2['retweet_mention'] = [dict_name[x] for x in retweet_network2['retweet_mention']]


# In[170]:

# final retweeet network dataset

retweet_network2


# In[171]:


# len(retweet_network2.loc[retweet_network2['retweet_mention'] == 'ManfredWeber'])


# In[172]:


import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
#from centralization import getCentralization
import networkx as nx
import networkx.algorithms.community as nxcom
from networkx.algorithms.community import greedy_modularity_communities
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams.update(plt.rcParamsDefault)
plt.rcParams.update({'figure.figsize': (15, 10)})
# get reproducible results
import random
from numpy import random as nprand
random.seed(123)
nprand.seed(123)


# In[173]:


G_w_directed =nx.from_pandas_edgelist(retweet_network2, source='retweet_mention', target='Twitter_handles', edge_attr='weight',create_using=nx.DiGraph())


# In[174]:


G_w_non_directed =nx.from_pandas_edgelist(retweet_network2, source='retweet_mention', target='Twitter_handles', edge_attr='weight')


# In[175]:


# edges = G_w.edges()

# #Drawing graph based on the information
# nx.draw_networkx(G_w, edgelist=edges, width=retweet_network2['weight'], with_labels = True)


# In[176]:


# nx.write_gexf(G_w,'retweet.gexf')


# In[ ]:





# # Get communities for non-directed network

# In[177]:


greedy_modularity_communities(G_w_non_directed)


# In[178]:


# Find the communities
communities = sorted(nxcom.greedy_modularity_communities(G_w_non_directed), key=len, reverse=True)
# Count the communities
print(len(communities))


# In[291]:


#communities[0]


# In[180]:


def set_node_community(G, communities):
    '''Add community to node attributes'''
    for c, v_c in enumerate(communities):
        for v in v_c:
            # Add 1 to save 0 for external edges
            G.nodes[v]['community'] = c + 1


# In[181]:


set_node_community(G_w_non_directed, communities)


# In[182]:


# f = retweet_network2[['Twitter_handles','Country']]
# f.loc[f['Twitter_handles'] == 'ManfredWeber']


# In[190]:


grouped_country = retweet_network2[['Twitter_handles','Country']].groupby("Twitter_handles")["Country"].agg("first")
dict_country = grouped_country.to_dict()
dict_country['manfred weber'] = 'Germany'
dict_country['andrius kubilius'] = 'Lithuania'
dict_country['iratxe garcía pérez'] = 'Spain'
dict_country['seán kelly'] = 'Ireland'
dict_country['reinhard bütikofer'] = 'Germany'
dict_country["sophia in 't veld"] = 'Netherlands'
dict_country['roberta metsola'] = 'Malta'

dict_country


# In[189]:


dict_name['RobertaMetsola']


# In[ ]:





# In[191]:


grouped_party = retweet_network2[['Twitter_handles','EU Party']].groupby("Twitter_handles")["EU Party"].agg("first")
dict_party = grouped_party.to_dict()
dict_party['manfred weber'] = "Group of the European People's Party (Christian Democrats)"
dict_party['andrius kubilius'] = "Group of the European People's Party (Christian Democrats)"
dict_party['iratxe garcía pérez'] = 'Group of the Progressive Alliance of Socialists and Democrats in the European Parliament'
dict_party['seán kelly'] = "Group of the European People's Party (Christian Democrats)"
dict_party['reinhard bütikofer'] = 'Group of the Greens/European Free Alliance'
dict_party["sophia in 't veld"] = 'Renew Europe Group'
dict_party['roberta metsola'] = "Group of the European People's Party (Christian Democrats)"
dict_party


# In[192]:


# retweet_network2[['Twitter_handles','Country']].groupby('Twitter_handles').Country.apply(list).to_dict()


# In[193]:


# retweet_network2[['Twitter_handles','Country']]


# In[225]:


nx.set_node_attributes(G_w_directed, dict_country, 'Country')


# In[226]:


nx.set_node_attributes(G_w_directed, dict_party, 'Party')


# In[227]:


nx.set_node_attributes(G_w_directed, nx.out_degree_centrality(G_w_directed), 'outdegree centrality')


# In[228]:


nx.set_node_attributes(G_w_directed, nx.in_degree_centrality(G_w_directed), 'indegree centrality')


# In[229]:


nx.set_node_attributes(G_w_directed, nx.closeness_centrality(G_w_directed.reverse()), 'outward closeness')


# In[230]:


nx.set_node_attributes(G_w_directed, nx.betweenness_centrality(G_w_directed), 'betweenness')


# In[231]:


nx.write_gexf(G_w_directed,'retweet_directed.gexf')




# In[232]:


nx.set_node_attributes(G_w_non_directed, dict_country, 'Country')


# In[233]:


nx.set_node_attributes(G_w_non_directed, dict_party, 'Party')


# In[234]:


nx.write_gexf(G_w_non_directed,'retweet_non_directed.gexf')



# In[46]:


#df.loc[df['Twitter_handles'] == 'RobertaMetsola']['EU Party'].values


# In[194]:


community_1 = list(communities[0])
community_1_country = [dict_country[x] for x in community_1]
community_1_party = [dict_party[x] for x in community_1]


# In[195]:


community_2 = list(communities[1])
community_2_country = [dict_country[x] for x in community_2]
community_2_party = [dict_party[x] for x in community_2]


# In[196]:


community_3 = list(communities[2])
community_3_country = [dict_country[x] for x in community_3]
community_3_party = [dict_party[x] for x in community_3]


# In[197]:


dict_country_1 = {}
country_set_1 = set(community_1_country) 
for country in country_set_1:
    dict_country_1[country] = community_1_country.count(country)
    
dict_country_1_sort = dict(sorted(dict_country_1.items(), key=lambda item: item[1], reverse=True))
    
a = pd.DataFrame(dict_country_1_sort.items())


# In[198]:


dict_country_2 = {}
country_set_2 = set(community_2_country) 
for country in country_set_2:
    dict_country_2[country] = community_2_country.count(country)
    
dict_country_2_sort = dict(sorted(dict_country_2.items(), key=lambda item: item[1], reverse=True))
    
b = pd.DataFrame(dict_country_2_sort.items())


# In[199]:


dict_country_3 = {}
country_set_3 = set(community_3_country) 
for country in country_set_3:
    dict_country_3[country] = community_3_country.count(country)
    
dict_country_3_sort = dict(sorted(dict_country_3.items(), key=lambda item: item[1], reverse=True))
    
c = pd.DataFrame(dict_country_3_sort.items())


# In[200]:


# dict_country = {'community_1': dict_country_1_sort,
#                'community_2': dict_country_2_sort,
#                'community_3': dict_country_3_sort}
# dict_country 


# In[201]:


country_community = pd.concat([a,b,c],axis=1)
country_community.columns = ['','community_1','','community_2','','community_3']
country_community


# In[202]:


country_community.to_excel('country_community.xlsx')



# In[203]:


dict_party_1 = {}
party_set_1 = set(community_1_party) 
for party in party_set_1:
    dict_party_1[party] = community_1_party.count(party)
    
dict_party_1_sort = dict(sorted(dict_party_1.items(), key=lambda item: item[1], reverse=True))
    
d = pd.DataFrame(dict_party_1_sort.items())


# In[204]:


dict_party_2 = {}
party_set_2 = set(community_2_party) 
for party in party_set_2:
    dict_party_2[party] = community_2_party.count(party)
    
dict_party_2_sort = dict(sorted(dict_party_2.items(), key=lambda item: item[1], reverse=True))
    
e = pd.DataFrame(dict_party_2_sort.items())


# In[205]:


dict_party_3 = {}
party_set_3 = set(community_3_party) 
for party in party_set_3:
    dict_party_3[party] = community_3_party.count(party)
    
dict_party_3_sort = dict(sorted(dict_party_3.items(), key=lambda item: item[1], reverse=True))
    
f = pd.DataFrame(dict_party_3_sort.items())


# In[206]:


# dict_country = {'community_1': dict_country_1_sort,
#                'community_2': dict_country_2_sort,
#                'community_3': dict_country_3_sort}
# dict_country 


# In[207]:


party_community = pd.concat([d,e,f],axis=1)
party_community.columns = ['','community_1','','community_2','','community_3']
party_community


# In[208]:


party_community.to_excel('party_community.xlsx')



# In[209]:


outdegree = pd.DataFrame(dict(nx.out_degree_centrality(G_w_directed)).items())
outdegree = outdegree.sort_values(1,ascending = False)


# In[292]:


#outdegree


# In[293]:


#outdegree[outdegree[1] == 0]


# In[211]:


indegree = pd.DataFrame(dict(nx.in_degree_centrality(G_w_directed)).items())
indegree = indegree.sort_values(1,ascending = False)


# In[294]:


#indegree[indegree[1] == 0]


# In[213]:


# outward closeness - with reverse()
closeness = pd.DataFrame(dict(nx.closeness_centrality(G_w_directed.reverse())).items())
closeness = closeness.sort_values(1,ascending = False)


# In[295]:


#closeness


# In[215]:


betweenness = pd.DataFrame(dict(nx.betweenness_centrality(G_w_directed)).items())
betweenness = betweenness.sort_values(1,ascending = False)


# In[296]:


#betweenness


# In[217]:


centrality_df = pd.concat([outdegree,indegree,closeness,betweenness],axis=1)
centrality_df.columns = ['','outdegree','','indegree','','closeness','','betweenness']
centrality_df


# In[297]:


#centrality_df.to_excel('centrality_df.xlsx')


# In[219]:


party_outdegree = outdegree.copy()
party_outdegree[0] = [dict_party[x] for x in party_outdegree[0]]


# In[298]:


#party_outdegree


# In[221]:


country_outdegree = outdegree.copy()
country_outdegree[0] = [dict_country[x] for x in country_outdegree[0]]


# In[299]:


#country_outdegree


# In[223]:


country_party_df = pd.concat([party_outdegree,country_outdegree],axis=1)
country_party_df.columns = ['','party_outdegree','','country_outdegree']
country_party_df


# In[224]:


country_party_df.to_excel('country_party.xlsx')


# In[158]:


df[df['Twitter_handles'] == 'angelika niebler']['Name']


# In[236]:


abc = df[df['Name'].str.contains('guy verhofstadt|iratxe garcía pérez|seán kelly|frances fitzgerald|jacek saryusz-wolski')]

# guy verhofstadt	0.728302
# 12	iratxe garcía pérez	0.313208
# 13	seán kelly	0.196226
# 10	frances fitzgerald	0.192453
# 20	jacek saryusz-wolski	0.173585


# In[300]:


#abc.Name.values


# In[301]:


#df_retweet_eu


# In[242]:


retweet_network2. rename(columns = {'Twitter_handles':'Retweet member', 'retweet_mention':'Retweeted member'}, inplace = True)

retweet_network2


# In[243]:


qwe = [x for x in retweet_network2['Retweet member']]


# In[244]:


for x in retweet_network2['Retweeted member']:
    qwe.append(x)


# In[251]:


len(set(qwe))


# In[275]:


len(retweet_network2)


# In[269]:


party123 = [dict_party[x] for x in set(qwe)]
party123


# In[270]:


dfcasc = pd.DataFrame(party123)

dfcasc['count'] = 1



new = dfcasc.groupby(0).sum()


# In[272]:


new2 = new.sort_values('count',ascending = False)
new2


# In[273]:


new2.to_excel('party_dis.xlsx')






