#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
project_name = "reco-tut-itr"; branch = "main"; account = "sparsh-ai"
project_path = os.path.join('/content', project_name)

if not os.path.exists(project_path):
    get_ipython().system(u'cp /content/drive/MyDrive/mykeys.py /content')
    import mykeys
    get_ipython().system(u'rm /content/mykeys.py')
    path = "/content/" + project_name; 
    get_ipython().system(u'mkdir "{path}"')
    get_ipython().magic(u'cd "{path}"')
    import sys; sys.path.append(path)
    get_ipython().system(u'git config --global user.email "recotut@recohut.com"')
    get_ipython().system(u'git config --global user.name  "reco-tut"')
    get_ipython().system(u'git init')
    get_ipython().system(u'git remote add origin https://"{mykeys.git_token}":x-oauth-basic@github.com/"{account}"/"{project_name}".git')
    get_ipython().system(u'git pull origin "{branch}"')
    get_ipython().system(u'git checkout main')
else:
    get_ipython().magic(u'cd "{project_path}"')


# In[31]:


import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().magic(u'matplotlib inline')


# In[5]:


files = glob.glob('./data/bronze/*')
files


# In[12]:


df1 = pd.read_parquet(files[0])
df1.head()


# In[13]:


df1.info()


# > Notes
# - There are total 2890 records but rating info for only 1777 is availble. We will keep only not-null rating records.
# - userId and itemId need to be categorical.
# - timestamp data type correction.

# In[14]:


df1 = df1.dropna(subset=['rating'])
df1.info()


# In[16]:


df1 = df1.astype({'userId': 'str', 'itemId': 'str'})
df1.info()


# In[26]:


df1['timestamp'] = pd.to_datetime(df1['timestamp'], unit='s')
df1.info()


# In[30]:


df1.describe(include='all', datetime_is_numeric=True).T


# > Notes
# - There are only 10 users providing ratings for 289 visiting places
# - Mean rating is 3.5
# - Data looks pretty old (year 1997-98)
# - Timespan is of 8 months (almost)

# In[34]:


fig, ax = plt.subplots(figsize=(16,4))
df1.groupby(df1['timestamp'])['rating'].count().plot(kind='line', ax=ax)
plt.show()


# In[38]:


fig, ax = plt.subplots(figsize=(8,4))
df1.rating.value_counts().plot(kind='bar', ax=ax)
plt.show()


# In[40]:


df1.rating.value_counts()


# In[41]:


df2 = pd.read_parquet(files[1])
df2.head()


# In[42]:


df2.info()


# > Notes
# - We have ratings for 289 items but here only 286 items are available, so we need to investigate and correct this mismatch
# - Also correct the itemId data type here also
# - No missing values, quite strange but ok

# In[43]:


df2 = df2.astype({'itemId': 'str'})


# In[44]:


items_df1 = df1.itemId.unique()
items_df2 = df2.itemId.unique()

set(items_df1) - set(items_df2)


# > Notes
# - Since we do not have metadata for these three items, let's see how many ratings are there for these, if not much, we can remove the records, otherwise, we will remove later if we train any hybrid model that used both rating and item metadata information.

# In[46]:


df1[df1.itemId.isin(list(set(items_df1) - set(items_df2)))].shape


# > Notes
# - 19 out of 1777, let's remove it

# In[47]:


df1 = df1[~df1.itemId.isin(list(set(items_df1) - set(items_df2)))]
df1.shape


# In[49]:


df2.describe().T


# In[50]:


df2.describe(include='O').T


# > Notes
# - Seems like creator of this dataset already preprocessed some fields, created one-hot encodings. We will remove these columns, to make things a little less messy and will do this type of encoding during modeling data preparation.

# In[51]:


df2 = df2.loc[:, ~df2.columns.str.startswith('travel_')]
df2 = df2.loc[:, ~df2.columns.str.startswith('religion_')]
df2 = df2.loc[:, ~df2.columns.str.startswith('season_')]
df2.info()


# In[53]:


get_ipython().system(u'mkdir ./data/silver')
df1.to_parquet('./data/silver/rating.parquet.gz', compression='gzip')
df2.to_parquet('./data/silver/items.parquet.gz', compression='gzip')


# In[54]:


get_ipython().system(u'git status')


# In[55]:


get_ipython().system(u"git add . && git commit -m 'commit' && git push origin main")

