#!/usr/bin/env python
# coding: utf-8

# > Note: KNN is a memory-based model, that means it will memorize the patterns and not generalize. It is simple yet powerful technique and compete with SOTA models like BERT4Rec.

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


# In[2]:


import os
import numpy as np
import pandas as pd
import scipy.sparse
from scipy.spatial.distance import correlation


# In[13]:


df = pd.read_parquet('./data/silver/rating.parquet.gz')
df.info()


# In[16]:


df2 = pd.read_parquet('./data/silver/items.parquet.gz')
df2.info()


# In[17]:


df = pd.merge(df, df2, on='itemId')
df.info()


# In[5]:


rating_matrix = pd.pivot_table(df, values='rating',
                               index=['userId'], columns=['itemId'])
rating_matrix


# In[6]:


def similarity(user1, user2):
    try:
        user1=np.array(user1)-np.nanmean(user1)
        user2=np.array(user2)-np.nanmean(user2)
        commonItemIds=[i for i in range(len(user1)) if user1[i]>0 and user2[i]>0]
        if len(commonItemIds)==0:
           return 0
        else:
           user1=np.array([user1[i] for i in commonItemIds])
           user2=np.array([user2[i] for i in commonItemIds])
           return correlation(user1,user2)
    except ZeroDivisionError:
        print("You can't divide by zero!")


# In[31]:


def nearestNeighbourRatings(activeUser, K):
    try:
        similarityMatrix=pd.DataFrame(index=rating_matrix.index,columns=['Similarity'])
        for i in rating_matrix.index:
            similarityMatrix.loc[i]=similarity(rating_matrix.loc[activeUser],rating_matrix.loc[i])
        similarityMatrix=pd.DataFrame.sort_values(similarityMatrix,['Similarity'],ascending=[0])
        nearestNeighbours=similarityMatrix[:K]
        neighbourItemRatings=rating_matrix.loc[nearestNeighbours.index]
        predictItemRating=pd.DataFrame(index=rating_matrix.columns, columns=['Rating'])
        for i in rating_matrix.columns:
            predictedRating=np.nanmean(rating_matrix.loc[activeUser])
            for j in neighbourItemRatings.index:
                if rating_matrix.loc[j,i]>0:
                   predictedRating += (rating_matrix.loc[j,i]-np.nanmean(rating_matrix.loc[j]))*nearestNeighbours.loc[j,'Similarity']
                predictItemRating.loc[i,'Rating']=predictedRating
    except ZeroDivisionError:
        print("You can't divide by zero!")            
    return predictItemRating


# In[36]:


def topNRecommendations(activeUser, N):
    try:
        predictItemRating = nearestNeighbourRatings(activeUser,N)
        placeAlreadyWatched = list(rating_matrix.loc[activeUser].loc[rating_matrix.loc[activeUser]>0].index)
        predictItemRating = predictItemRating.drop(placeAlreadyWatched)
        topRecommendations = pd.DataFrame.sort_values(predictItemRating,['Rating'],ascending = [0])[:N]
        topRecommendationTitles = (df.loc[df.itemId.isin(topRecommendations.index)])
    except ZeroDivisionError:
        print("You can't divide by zero!")
    return list([topRecommendationTitles.location,
                 topRecommendationTitles.place,
                 topRecommendationTitles.state,
                 topRecommendationTitles.location_rating])


# In[42]:


def favoritePlace(activeUser,N):
    topPlace=pd.DataFrame.sort_values(df[df.userId==activeUser],['rating'],ascending=[0])[:N]
    return list([topPlace.location,
                 topPlace.place,
                 topPlace.state,
                 topPlace.location_rating])


# In[37]:


activeUser = 4


# In[44]:


print("Your favorite places are: ")
fav_place=pd.DataFrame(favoritePlace(str(activeUser),4))
fav_place=fav_place.T
fav_place=fav_place.sort_values(by='location_rating', ascending=False)
fav_place


# In[45]:


print("The recommended places for you are: ")
topN = pd.DataFrame(topNRecommendations(str(activeUser), 4))
topN = topN.T
topN = topN.sort_values(by = 'location_rating', ascending=False).drop_duplicates().reset_index(drop=True)
topN

