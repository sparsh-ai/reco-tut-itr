#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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


# In[5]:


import pandas as pd


# In[6]:


ratings = pd.read_csv("./data/bronze/ratings.csv",encoding='ISO-8859-1')
ratings.head()


# In[7]:


ratings.info()


# In[8]:


ratings.describe()


# In[9]:


ratings.to_parquet('./data/bronze/ratings.parquet.gz', compression='gzip')


# In[10]:


items = pd.read_csv("./data/bronze/items.csv",encoding='ISO-8859-1')
items.head()


# In[12]:


items.info()


# In[14]:


items.describe().T


# In[15]:


items.to_parquet('./data/bronze/items.parquet.gz', compression='gzip')


# In[20]:


get_ipython().system(u'mv ./data/*.gz ./data/bronze')


# In[21]:


get_ipython().system(u'git status')


# In[22]:


get_ipython().system(u"git add . && git commit -m 'commit' && git push origin main")

