#!/usr/bin/env python
# coding: utf-8

# In[1]:


#pip install transformers


# In[7]:


pip install emoji==1.7.0


# In[8]:


#pip install ipywidgets==5.0.0


# In[10]:


pip install -q transformers tweepy wordcloud matplotlib


# In[2]:


import pandas as pd


# In[602]:


df_sanction = pd.read_pickle('df_for_sentiment.pkl')
#df_sanction


# In[4]:


sanction_corpus = [x for x in df_sanction['Tweet']]
# sanction_corpus


# In[5]:


from transformers import pipeline


# In[5]:


task='sentiment'
MODEL = f"cardiffnlp/twitter-roberta-base-{task}"


# In[72]:


sentiment_analysis = pipeline(model="nlptown/bert-base-multilingual-uncased-sentiment")


# In[73]:


sentiment_analysis("“We don’t need your soldiers, boots on the ground.\n\nBut we need fuel, for our vehicles.\n\nWe need blood, for our wounded.\n\nAnd we need stronger sanctions on Putin, to force him to stop his aggression.”")


# In[6]:


import tqdm


# In[14]:


# Set up the inference pipeline using a model from the 🤗 Hubb
 
# Let's run the sentiment analysis on each tweet
tweets = []
for tweet in tqdm.tqdm(sanction_corpus):
   try:
     sentiment = sentiment_analysis(tweet)
     tweets.append({'tweet': tweet, 'sentiment': sentiment[0]['label']})
   except:
     pass


# In[ ]:





# In[23]:


df_sentiment = pd.DataFrame(tweets)
pd.set_option('display.max_colwidth', None)


# In[603]:


#df_sentiment


# In[25]:


df_sentiment.to_pickle('sentiment.pkl')


# In[ ]:





# In[604]:


#df_sentiment[df_sentiment["sentiment"] == '5 stars']


# In[ ]:



# In[71]:


sentiment_analysis_2 = pipeline(model="distilbert-base-uncased-finetuned-sst-2-english")


# In[74]:


sentiment_analysis_2("“We don’t need your soldiers, boots on the ground.\n\nBut we need fuel, for our vehicles.\n\nWe need blood, for our wounded.\n\nAnd we need stronger sanctions on Putin, to force him to stop his aggression.”")


# In[75]:


# Set up the inference pipeline using a model from the 🤗 Hubb
 
# Let's run the sentiment analysis on each tweet
tweets_2 = []
for tweet in tqdm.tqdm(sanction_corpus):
   try:
     sentiment = sentiment_analysis_2(tweet)
     tweets_2.append({'tweet': tweet, 'sentiment': sentiment[0]['label']})
   except:
     pass


# In[ ]:





# In[ ]:





# In[398]:


sentiment_analysis_3 = pipeline(model="cardiffnlp/twitter-roberta-base-sentiment-latest")


# In[403]:


sentiment_analysis_3("  @IratxeGarper: The @PES_PSE condemns Russia’s military attack against Ukraine and strongly supports tough EU sanctions on Russia.  Toda…")


# In[404]:


tweets_3 = []
for tweet in tqdm.tqdm(sanction_corpus):
   try:
     sentiment = sentiment_analysis_3(tweet)
     tweets_3.append({'tweet': tweet, 'sentiment': sentiment[0]['label']})
   except:
     pass


# In[ ]:



# In[76]:


df_sentiment_2 = pd.DataFrame(tweets_2)
pd.set_option('display.max_colwidth', None)


# In[605]:


#df_sentiment_2


# In[89]:


df_sentiment_2.to_pickle('sentiment_2.pkl')


# In[ ]:



# In[405]:


df_sentiment_3 = pd.DataFrame(tweets_3)


# In[606]:


#df_sentiment_3


# In[ ]:



# In[412]:


import seaborn as sns
import matplotlib.pyplot as plt

# count the number of tweets by sentiments
sentiment_counts = df_sentiment_3.groupby(['sentiment']).size()
print(sentiment_counts)

# visualize the sentiments
fig = plt.figure(figsize=(6,6), dpi=100)
ax = plt.subplot(111)
explode = (0.05,0.05,0.2)
sentiment_counts.plot.pie(ax=ax, explode=explode, shadow=True, autopct='%1.1f%%', startangle=270, fontsize=12, label="")


# In[ ]:


# Plot with autopct

colors=['rosybrown', 'moccasin', 'lightyellow', 'darkseagreen','salmon','palegreen','skyblue','silver'] #,'plum'
explode = (0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1)

plt.figure(0)
plt.figure(figsize=(8,10))
plt.title('Distribution of "likes" by EU Party', fontweight='bold')
plt.pie(party_df.fav_by_party, explode=explode,shadow=True,labels=party_df.index, autopct='%.1f%%', wedgeprops={'edgecolor':'maroon','alpha':0.5}, colors=colors)
#plt.savefig('Save Pie Chart.png')

# Create second chart.

plt.figure(1)
plt.figure(figsize=(8,10))
plt.title('Distribution of EU Party retweets', fontweight='bold')
plt.pie(party_df.retw_by_party, explode=explode, shadow=True, labels=party_df.index, autopct='%.1f%%', wedgeprops={'edgecolor':'maroon','alpha':0.5}, colors=colors) 
#plt.savefig('Save Pie Chart.png')

# Create third chart.

plt.figure(2)
plt.figure(figsize=(8,10))
plt.title('Distribution of times EU Party is retweeted', fontweight='bold')
plt.pie(party_df.retwed_by_party, explode=explode, shadow=True, labels=party_df.index, autopct='%.1f%%', wedgeprops={'edgecolor':'maroon','alpha':0.5}, colors=colors) 
#plt.savefig('Save Pie Chart.png')

plt.show()


# In[426]:


from wordcloud import WordCloud
from wordcloud import STOPWORDS
 
    
# Wordcloud with negative tweets
negative_tweets = df_sentiment_3[df_sentiment_3["sentiment"] == 'Negative']['tweet']
stop_words = ["https", "co", "RT"] + list(STOPWORDS)
negative_wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white", stopwords = stop_words).generate(str(negative_tweets))
plt.figure()
plt.title("Negative Tweets - Wordcloud")
plt.imshow(negative_wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()
    
# Wordcloud with positive tweets
positive_tweets = df_sentiment_3[df_sentiment_3["sentiment"] == 'Positive']['tweet']
stop_words = ["https", "co", "RT"] + list(STOPWORDS)
positive_wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white", stopwords = stop_words).generate(str(positive_tweets))
plt.figure()
plt.title("Positive Tweets - Wordcloud")
plt.imshow(positive_wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()


# In[ ]:





# In[427]:


sentiment_list = [x['sentiment'] for x in tweets_3]
sentiment_list


# In[428]:


df_sanction['sentiment'] = sentiment_list


# In[607]:


#df_sanction


# In[430]:


df_sanction.columns


# In[445]:


df_sanction[df_sanction['key_entities_classifier'] == True].groupby(['sentiment']).size()


# In[444]:


# count the number of tweets by sentiments
sentiment_counts_energy = df_sanction[df_sanction['energy_classifier'] == True].groupby(['sentiment']).size()
sentiment_counts_finance = df_sanction[df_sanction['finance_classifier'] == True].groupby(['sentiment']).size()
sentiment_counts_oligarch = df_sanction[df_sanction['oligarch_classifier'] == True].groupby(['sentiment']).size()
sentiment_counts_country = df_sanction[df_sanction['countries_classifier'] == True].groupby(['sentiment']).size()
sentiment_counts_transport = df_sanction[df_sanction['transport_classifier'] == True].groupby(['sentiment']).size()
sentiment_counts_media = df_sanction[df_sanction['media_classifier'] == True].groupby(['sentiment']).size()
sentiment_counts_diplomatic = df_sanction[df_sanction['diplomatic_classifier'] == True].groupby(['sentiment']).size()
sentiment_counts_trade = df_sanction[df_sanction['trade_classifier'] == True].groupby(['sentiment']).size()
sentiment_counts_ru_state = df_sanction[df_sanction['ru_state_classifier'] == True].groupby(['sentiment']).size()
sentiment_counts_key_entities = df_sanction[df_sanction['key_entities_classifier'] == True].groupby(['sentiment']).size()

# print(sentiment_counts_energy)
# print(sentiment_counts_finance)
# print(sentiment_counts_oligarch)
# print(sentiment_counts_country)

# visualize the sentiments

fig,ax = plt.subplots(3,4,figsize=(16,10))
# pie(sentiment_counts_energy, autopct='%1.1f%%', startangle=270,ax=ax[0,0])
explode = (0.05,0.05,0.2)
explode1 = (0.05,0.05)
ax[0,0].pie(sentiment_counts_energy,labels = ['Negative','Neutral', 'Positive'], explode=explode, shadow=True, autopct='%1.0f%%')
ax[0,1].pie(sentiment_counts_finance,labels = ['Negative', 'Neutral','Positive'], explode=explode, shadow=True, autopct='%1.0f%%')
ax[0,2].pie(sentiment_counts_oligarch,labels = ['Negative', 'Neutral','Positive'], explode=explode, shadow=True, autopct='%1.0f%%')
ax[0,3].pie(sentiment_counts_country,labels = ['Negative', 'Neutral','Positive'], explode=explode, shadow=True, autopct='%1.0f%%')
ax[1,0].pie(sentiment_counts_transport,labels = ['Negative','Neutral'], explode=explode1, shadow=True, autopct='%1.0f%%')
ax[1,1].pie(sentiment_counts_media,labels = ['Negative','Neutral', 'Positive'], explode=explode, shadow=True, autopct='%1.0f%%')
ax[1,2].pie(sentiment_counts_diplomatic,labels = ['Negative', 'Neutral'], explode=explode1, shadow=True, autopct='%1.0f%%')
ax[1,3].pie(sentiment_counts_trade,labels = ['Negative','Neutral', 'Positive'], explode=explode, shadow=True, autopct='%1.0f%%')
ax[2,0].pie(sentiment_counts_ru_state,labels = ['Negative', 'Neutral','Positive'], explode=explode, shadow=True, autopct='%1.0f%%')
ax[2,1].pie(sentiment_counts_key_entities,labels = ['Negative', 'Neutral','Positive'], explode=explode, shadow=True, autopct='%1.0f%%')


ax[0,0].set_title('energy')
ax[0,1].set_title('finance')
ax[0,2].set_title('oligarch')
ax[0,3].set_title('country')
ax[1,0].set_title('transport')
ax[1,1].set_title('media')
ax[1,2].set_title('diplomatic')
ax[1,3].set_title('trade')
ax[2,0].set_title('ru_state')
ax[2,1].set_title('key_entities')

ax[2,2].set_axis_off()
ax[2,3].set_axis_off()

plt.show()


# In[446]:


sentiment_counts_energy


# In[447]:


negative_tweets = df_sanction[df_sanction['sentiment'] == 'Negative'][['energy_classifier','transport_classifier', 
       'finance_classifier','media_classifier','diplomatic_classifier','trade_classifier',
       'ru_state_classifier', 'oligarch_classifier','key_entities_classifier','countries_classifier']]


# In[608]:


#negative_tweets


# In[449]:


a = negative_tweets.sum(axis = 'rows').to_dict()


# In[450]:


negative_distribution = pd.DataFrame.from_dict(a, orient='index',columns=['Count'])
negative_distribution = negative_distribution.sort_values('Count',ascending = True)
negative_distribution


# In[451]:


fig, ax = plt.subplots()
bars = ax.barh(list(negative_distribution.index.values), negative_distribution['Count'])
ax.bar_label(bars)
plt.show()


# In[455]:


positive_tweets = df_sanction[df_sanction['sentiment'] == 'Positive'][['energy_classifier','transport_classifier', 
       'finance_classifier','media_classifier','diplomatic_classifier','trade_classifier',
       'ru_state_classifier', 'oligarch_classifier','key_entities_classifier','countries_classifier']]


# In[456]:


b = positive_tweets.sum(axis = 'rows').to_dict()


# In[458]:


positive_distribution = pd.DataFrame.from_dict(b, orient='index',columns=['Count'])
positive_distribution = positive_distribution.sort_values('Count',ascending = True)
positive_distribution


# In[503]:


Neutral_tweets = df_sanction[df_sanction['sentiment'] == 'Neutral'][['energy_classifier','transport_classifier', 
       'finance_classifier','media_classifier','diplomatic_classifier','trade_classifier',
       'ru_state_classifier', 'oligarch_classifier','key_entities_classifier','countries_classifier']]


# In[504]:


k = Neutral_tweets.sum(axis = 'rows').to_dict()


# In[505]:


Neu_distribution = pd.DataFrame.from_dict(k, orient='index',columns=['Count'])
Neu_distribution = Neu_distribution.sort_values('Count',ascending = True)
Neu_distribution


# In[ ]:





# In[459]:


fig, ax = plt.subplots()
bars1 = ax.barh(list(positive_distribution.index.values), positive_distribution['Count'])
ax.bar_label(bars1)
plt.show()


# In[517]:


fig, ax = plt.subplots(3,1, figsize=(12,11))

bars = ax[0].barh(list(negative_distribution.index.values), negative_distribution['Count'])
ax[0].bar_label(bars)
ax[0].set_title('Negative tweets distribution')

bars1 = ax[1].barh(list(positive_distribution.index.values), positive_distribution['Count'])
ax[1].bar_label(bars1)
ax[1].set_title('Positive tweets distribution')

bars2 = ax[2].barh(list(Neu_distribution.index.values), Neu_distribution['Count'])
ax[2].bar_label(bars2)
ax[2].set_title('Neutral tweets distribution')

plt.show()


# In[461]:


import numpy as np


# In[462]:


# from transformers import AutoTokenizer, AutoModelForSequenceClassification

# tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

# model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")


# In[463]:


# inputs = tokenizer("STOP SWIFT! STOP RUSSIA!\nI call on the @EU_Commission to exclude Russia from the #SWIFT interbank system! \n\nRussia's excluding from the SWIFT is an urgent action that must be taken immediately!\n\nSolidarity with #Ukraine must be manifested beyond statements! #UkraineInvasion", return_tensors="pt")

# outputs = model(**inputs)


# In[ ]:




# In[464]:


#df_sanction


# In[465]:


df_retweet = df_sanction.loc[df_sanction['is_retweet'] == 1].copy()
df_retweet


# In[466]:


import re


# In[467]:


people = []
for x in df_retweet['Tweet']:
    s = re.findall("@([a-zA-Z0-9]{1,15})", x)
    people.append(s)


# In[468]:


people_retweet = []
for x in people:
    if x == []:
        people_retweet.append('')
    else:
        people_retweet.append(x[0])
        


# In[469]:


df_retweet['retweet_mention'] = people_retweet


# In[470]:


twitter_handle = set(df_retweet['Twitter_handles'])


# In[471]:


df_retweet_eu = df_retweet.loc[df_retweet['retweet_mention'].isin(twitter_handle)]


# In[472]:


df_retweet_eu


# In[473]:


name_df = df_sanction[['Name','Twitter_handles']]
grouped = name_df.groupby('Twitter_handles')['Name'].agg("first")
dict_name = grouped.to_dict()


# In[474]:


df_retweet_eu['Name'] = [dict_name[x] for x in df_retweet_eu['retweet_mention']]


# In[475]:


df_retweet_eu = df_retweet_eu.drop(['retweet_mention'],axis = 1)


# In[609]:


#df_retweet_eu


# In[610]:


df_non_retweet = df_sanction.loc[df_sanction['is_retweet'] == 0].copy()
# df_non_retweet


# In[478]:


df_clean = pd.concat([df_retweet_eu, df_non_retweet], join="inner")


# In[611]:


# df_clean



# In[480]:


negative_tweets_name = df_clean[df_clean['sentiment'] == 'Negative']['Name']
q = negative_tweets_name.value_counts().to_dict()
neg_name_distribution = pd.DataFrame.from_dict(q, orient='index',columns=['Count'])
neg_name_distribution = neg_name_distribution.sort_values('Count',ascending = True)
neg_name_distribution = neg_name_distribution.tail(10)


# In[481]:


positive_tweets_name = df_clean[df_clean['sentiment'] == 'Positive']['Name']
w = positive_tweets_name.value_counts().to_dict()
pos_name_distribution = pd.DataFrame.from_dict(w, orient='index',columns=['Count'])
pos_name_distribution = pos_name_distribution.sort_values('Count',ascending = True)

pos_name_distribution = pos_name_distribution.tail(10)


# In[484]:


Neutral_tweets_name = df_clean[df_clean['sentiment'] == 'Neutral']['Name']
p = Neutral_tweets_name.value_counts().to_dict()
Neutral_name_distribution = pd.DataFrame.from_dict(p, orient='index',columns=['Count'])
Neutral_name_distribution = Neutral_name_distribution.sort_values('Count',ascending = True)

Neutral_name_distribution = Neutral_name_distribution.tail(10)


# In[485]:


# fig, ax = plt.subplots()
# bars3 = ax.barh(list(positive_distribution.index.values), positive_distribution['Count'])
# ax.bar_label(bars2)
# plt.show()


# In[518]:


fig, ax = plt.subplots(3,1, figsize=(12,11))

bars3 = ax[0].barh(list(neg_name_distribution.index.values), neg_name_distribution['Count'])
ax[0].bar_label(bars3)
ax[0].set_title('Negative tweets distribution')

bars4 = ax[1].barh(list(pos_name_distribution.index.values), pos_name_distribution['Count'])
ax[1].bar_label(bars4)
ax[1].set_title('Positive tweets distribution')

bars5 = ax[2].barh(list(Neutral_name_distribution.index.values), Neutral_name_distribution['Count'])
ax[2].bar_label(bars5)
ax[2].set_title('Neutral tweets distribution')

plt.show()



# In[487]:


negative_tweets_party = df_clean[df_clean['sentiment'] == 'Negative']['EU Party']
y = negative_tweets_party.value_counts().to_dict()
neg_party_distribution = pd.DataFrame.from_dict(y, orient='index',columns=['Count'])
neg_party_distribution = neg_party_distribution.sort_values('Count',ascending = False)
#neg_party_distribution = neg_party_distribution.tail(10)


# In[488]:


neg_party_distribution


# In[380]:


#neg_party_distribution = neg_party_distribution.to_num


# In[489]:


positive_tweets_party = df_clean[df_clean['sentiment'] == 'Positive']['EU Party']
u = positive_tweets_party.value_counts().to_dict()
pos_party_distribution = pd.DataFrame.from_dict(u, orient='index',columns=['Count'])
pos_party_distribution = pos_party_distribution.sort_values('Count',ascending = False)
#pos_party_distribution = pos_party_distribution.to_numpy
#pos_party_distribution = pos_name_distribution.tail(10)


# In[491]:


Neu_tweets_party = df_clean[df_clean['sentiment'] == 'Neutral']['EU Party']
o = Neu_tweets_party.value_counts().to_dict()
Neu_party_distribution = pd.DataFrame.from_dict(o, orient='index',columns=['Count'])
Neu_party_distribution = Neu_party_distribution.sort_values('Count',ascending = False)
#pos_party_distribution = pos_party_distribution.to_numpy
#pos_party_distribution = pos_name_distribution.tail(10)


# In[501]:


fig, ax = plt.subplots(3,1, figsize=(16,12))

explode = (0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1)

ax[0].pie(neg_party_distribution['Count'],labels = list(neg_party_distribution.index.values), explode = explode, shadow=True, autopct='%1.0f%%')

ax[0].set_title('Negative tweets distribution')

ax[1].pie(pos_party_distribution['Count'],labels = list(pos_party_distribution.index.values), explode = explode, shadow=True, autopct='%1.0f%%')

ax[1].set_title('Positive tweets distribution')

ax[2].pie(Neu_party_distribution['Count'],labels = list(Neu_party_distribution.index.values), explode = explode, shadow=True, autopct='%1.0f%%')

ax[2].set_title('Neutral tweets distribution')

plt.show()


# In[ ]:





# In[601]:


fig, ax = plt.subplots(1,3, figsize=(20,12))

bars20 = ax[0].bar(list(neg_party_distribution.index.values), neg_party_distribution['Count'])
ax[0].bar_label(bars20)
ax[0].set_title('Negative tweets distribution',fontsize=15)

bars21 = ax[1].bar(list(pos_party_distribution.index.values), pos_party_distribution['Count'])
ax[1].bar_label(bars21)
ax[1].set_title('Positive tweets distribution',fontsize=15)

bars22 = ax[2].bar(list(Neu_party_distribution.index.values), Neu_party_distribution['Count'])
ax[2].bar_label(bars22)
ax[2].set_title('Neutral tweets distribution',fontsize=15)

# ax[0].set_xticklabels(list(neg_party_distribution.index.values),rotation=45)
# ax[1].set_xticklabels(list(neg_party_distribution.index.values),rotation=45)
# ax[2].set_xticklabels(list(neg_party_distribution.index.values),rotation=45)

fig.autofmt_xdate(rotation= 70)

ax[0].tick_params(labelsize=15)
ax[1].tick_params(labelsize=15)
ax[2].tick_params(labelsize=15)

plt.show()
plt.tight_layout()


# In[ ]:



# In[509]:


sample = df_sentiment_3.sample(n=400, random_state=1)


# In[414]:


sample.to_excel('sentiment data.xlsx',index=False)


# In[510]:


df1, df2, df3 = np.array_split(sample, 3)


# In[511]:


for i, df in enumerate(np.array_split(sample, 3)):
    df.to_excel(f"sentiment 3 data{i+1}.xlsx", index=False)


# In[ ]:



# In[ ]:


guy verhofstadt	0.728302
12	iratxe garcía pérez	0.313208
13	seán kelly	0.196226
10	frances fitzgerald	0.192453
20	jacek saryusz-wolski	0.173585


# In[520]:


guy_data = df_clean.loc[df_clean['Name'] == 'guy verhofstadt']


# In[521]:


guy_data = guy_data[['energy_classifier','transport_classifier', 
       'finance_classifier','media_classifier','diplomatic_classifier','trade_classifier',
       'ru_state_classifier', 'oligarch_classifier','key_entities_classifier','countries_classifier']]


# In[612]:


#guy_data


# In[529]:


guy_data.sum()


# In[ ]:





# In[534]:


chris_De_data = df_clean.loc[df_clean['EU Party'] == "Group of the European People's Party (Christian Democrats)"]


# In[535]:


chris_De_data = chris_De_data[['energy_classifier','transport_classifier', 
       'finance_classifier','media_classifier','diplomatic_classifier','trade_classifier',
       'ru_state_classifier', 'oligarch_classifier','key_entities_classifier','countries_classifier']]


# In[536]:


chris_De_data.sum()


# In[ ]:





# In[538]:


top5_actor_data = df_clean.loc[(df_clean['Name'] == 'guy verhofstadt') | (df_clean['Name'] == 'iratxe garcía pérez')
                              | (df_clean['Name'] == 'seán kelly') | (df_clean['Name'] == 'frances fitzgerald')
                              | (df_clean['Name'] == 'jacek saryusz-wolski')]


# In[613]:


#top5_actor_data


# In[541]:


top5_actor_data = top5_actor_data[['energy_classifier','transport_classifier', 
       'finance_classifier','media_classifier','diplomatic_classifier','trade_classifier',
       'ru_state_classifier', 'oligarch_classifier','key_entities_classifier','countries_classifier']]


# In[564]:


top5_actor_data.sum()


# In[563]:


fig, ax = plt.subplots(3,1, figsize=(12,11))

bars10 = ax[0].barh(guy_data.sum().sort_values().index.values, guy_data.sum().sort_values().values)
ax[0].bar_label(bars10)
ax[0].set_title('Topic mentioned distribution by Guy Verhofstadt')

bars11 = ax[1].barh(top5_actor_data.sum().sort_values().index.values, top5_actor_data.sum().sort_values().values)
ax[1].bar_label(bars11)
ax[1].set_title('Topic mentioned distribution by Top 5 influencial actors')

bars12 = ax[2].barh(chris_De_data.sum().sort_values().index.values, chris_De_data.sum().sort_values().values)
ax[2].bar_label(bars12)
ax[2].set_title("Topic mentioned distribution by Group of the European People's Party (Christian Democrats)")

plt.show()
