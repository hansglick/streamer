#!/usr/bin/env python
# coding: utf-8

# %load_ext autoreload
# %autoreload 2

# In[1]:


import pandas as pd
import json
import pickle
import os
import itertools
from ast import literal_eval
import numpy as np
import sys
from datetime import datetime
from IPython.display import clear_output, display
import subprocess
from datetime import datetime
pd.options.display.float_format = '{:.0f}'.format


# In[2]:


def LoadJsonFile(filename): 
    with open(filename, 'r') as f:
        DicConfig = json.load(f)
    return DicConfig


def GlobalDicDeplier(OneDic):
    for k,v in OneDic.items():
        exec('globals()[k] = v')
    return None


# In[3]:


DicConfig = LoadJsonFile(os.path.join(os.getcwd(),"config.json"))
GlobalDicDeplier(DicConfig)
sys.path.append(Root)
from fun import *
print("Load Config variables")


# In[4]:


path = os.path.join(Root,FolderProject,"RefRT.pkl")
RefRT = LoadPickleOrInit(path)

path = os.path.join(Root,FolderProject,"RefFam.pkl")
RefFam = LoadPickleOrInit(path)

path = os.path.join(Root,FolderProject,"RefInf.pkl")
RefInf = LoadPickleOrInit(path)


path = os.path.join(Root,FolderProject,"BatchRT.pkl")
BatchRT = LoadPickleOrInit(path)

path = os.path.join(Root,FolderProject,"BatchFamousTweet.pkl")
BatchFam = LoadPickleOrInit(path)

path = os.path.join(Root,FolderProject,"BatchInf.pkl")
BatchInf = LoadPickleOrInit(path)

print("Load Dataframes")


# # RT Part

# RefRT = RefRT.append(BatchRT,ignore_index = True)

# In[5]:


RefRT = pd.concat([RefRT,BatchRT],axis=0,sort=True)
RefRT[["status"]] = RefRT[["status"]].fillna(value="ko")
RefRT.reset_index(inplace = True,drop = True)
RefRT.drop_duplicates(subset=["TWEETID","USERID"],inplace=True)

PickleDump(os.path.join(Root,FolderProject,"RefRT.pkl"),RefRT)

path = os.path.join(Root,FolderProject,"RefRT.pkl")

RefRT_memory = RetrieveSize(path)
RefRT_rows = len(RefRT)
RefRT_tweets = len(np.unique(RefRT.AUTHORTWEETID))
RefRT_users = len(np.unique(RefRT.USERID))
RefRT_authors = len(np.unique(RefRT.AUTHORID))
RefRT_datemin = RefRT.TWEETUNIXEPOCH.min()
RefRT_datemax = RefRT.TWEETUNIXEPOCH.max()

print("RT Part")


# # Fam Part

# In[6]:


RefFam = pd.concat([RefFam,BatchFam],axis=0,sort=True)
RefFam[["status"]] = RefFam[["status"]].fillna(value="ko")
RefFam.reset_index(inplace = True,drop = True)
RefFam.drop_duplicates(inplace=True)
RefFam.drop_duplicates(subset = "AUTHORTWEETID", inplace = True)

PickleDump(os.path.join(Root,FolderProject,"RefFam.pkl"),RefFam)

path = os.path.join(Root,FolderProject,"RefFam.pkl")

RefFam_memory = RetrieveSize(path)
RefFam_rows = len(RefFam)
RefFam_authors = len(np.unique(RefFam.AUTHORID))
RefFam_tweets = len(np.unique(RefFam.AUTHORTWEETID))
RefFam_datemin = RefFam.AUTHORTWEETUNIXEPOCH.min()
RefFam_datemax = RefFam.AUTHORTWEETUNIXEPOCH.max()

print("Fam Part")


# # Inf Part

# In[7]:


RefInf = RefInf.append(BatchInf,ignore_index = True)
RefInf.reset_index(inplace = True,drop = True)
RefInf = RefInf.groupby("AUTHORID").first().reset_index()
PickleDump(os.path.join(Root,FolderProject,"RefInf.pkl"),RefInf)

path = os.path.join(Root,FolderProject,"RefInf.pkl")

RefInf_memory = RetrieveSize(path)
RefInf_rows = len(RefInf)
RefInf_authors = len(np.unique(RefInf.AUTHORID))

print("Inf Part")


# # Logs

# In[8]:


def FormatNumber(Size):
    res = f'{Size:,}'
    return res


# In[9]:


RefInf_rows = FormatNumber(RefInf_rows)
RefInf_authors = FormatNumber(RefInf_authors)

RefFam_rows = FormatNumber(RefFam_rows)
RefFam_authors = FormatNumber(RefFam_authors)
RefFam_tweets = FormatNumber(RefFam_tweets)

RefRT_rows = FormatNumber(RefRT_rows)
RefRT_tweets = FormatNumber(RefRT_tweets)
RefRT_users = FormatNumber(RefRT_users)
RefRT_authors = FormatNumber(RefRT_authors)

RefFam_datemin = str(pd.to_datetime(RefFam_datemin,unit="s"))
RefFam_datemax = str(pd.to_datetime(RefFam_datemax,unit="s"))
RefRT_datemin = str(pd.to_datetime(RefRT_datemin,unit="s"))
RefRT_datemax = str(pd.to_datetime(RefRT_datemax,unit="s"))

print("Format values")


# In[10]:


RefInfDic = {"RefInf_memory":RefInf_memory,
"RefInf_rows":RefInf_rows,
"RefInf_authors":RefInf_authors}

RefFamDic = {"RefFam_memory" : RefFam_memory,
"RefFam_rows" : RefFam_rows,
"RefFam_authors" : RefFam_authors,
"RefFam_tweets" : RefFam_tweets,
"RefFam_datemin" : RefFam_datemin,
"RefFam_datemax" : RefFam_datemax}

RefRTDic = {"RefRT_memory" : RefRT_memory,
"RefRT_rows" : RefRT_rows,
"RefRT_tweets" : RefRT_tweets,
"RefRT_users" : RefRT_users,
"RefRT_authors" : RefRT_authors,
"RefRT_datemin" : RefRT_datemin,
"RefRT_datemax" : RefRT_datemax}

RefLogs = {"Fam":RefFamDic,
             "RT":RefRTDic,
             "Inf":RefInfDic}
filename = os.path.join(Root,FolderProject,"Ref.log")
AppendStringToFile(filename,RefLogs)

print("Write logs")


# In[13]:


RefFam.shape

