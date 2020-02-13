#!/usr/bin/env python
# coding: utf-8

# # ATTENTION!
# # A lancer depuis le Project Folder
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


tweetslist,errors = ReadTweetsToList(Root,FolderProject,TweetsFilename)
print("Nombre de tweets : ", str(len(tweetslist)))
print("errors : ",str(errors))
print("Read Tweets")


# In[5]:


# Clean Flow
RowsToRemove = len(tweetslist) + errors
commandbash = WriteCleaningScript(RowsToRemove,Root,FolderProject,TweetsFilename)
resultat = subprocess.call(commandbash,shell = True)
if resultat == 0:
    print(str(RowsToRemove),"Lignes correctement supprimées")
else:
    print("Problème lors du cleaning")


# In[6]:


f = "%a %b %d %H:%M:%S +0000 %Y"
df = pd.DataFrame(tweetslist)
df.TWEETTIMESTAMP = pd.to_datetime(df.TWEETTIMESTAMP,format = f, errors='ignore')
df["TWEETUNIXEPOCH"] = df.TWEETTIMESTAMP.map(lambda a : a.value // 10**9)
df.TWEETUNIXEPOCH = df.TWEETUNIXEPOCH + 3600
print("Convert tweets to dataframe")


# In[7]:


df = df.replace({'True':True,'False':False})
df.USERID = pd.to_numeric(df.USERID,errors="coerce")
df.USERFOLLOWERS = pd.to_numeric(df.USERFOLLOWERS,errors="coerce")
df.TWEETID = pd.to_numeric(df.TWEETID,errors="coerce")
df.AUTHORID = pd.to_numeric(df.AUTHORID,errors="coerce")
df.AUTHORFOLLOWERS = pd.to_numeric(df.AUTHORFOLLOWERS,errors="coerce")
df.AUTHORTWEETID = pd.to_numeric(df.AUTHORTWEETID,errors="coerce")
df.TWEETUNIXEPOCH = pd.to_numeric(df.TWEETUNIXEPOCH,errors="coerce")
print("Clean errors")


# In[8]:


dfrt = df.copy()[df.ISRETWEET]
dfrt.dropna(inplace=True)
dfrt.AUTHORTWEETTIMESTAMP = pd.to_datetime(dfrt.AUTHORTWEETTIMESTAMP,format = f, errors='ignore')
dfrt["AUTHORTWEETUNIXEPOCH"] = dfrt.AUTHORTWEETTIMESTAMP.map(lambda a : a.value // 10**9)
dfrt.AUTHORTWEETUNIXEPOCH = dfrt.AUTHORTWEETUNIXEPOCH + 3600


dfsimple = df.copy()[~df.ISRETWEET]
vartoremove = [ 'AUTHORNAME',
 'AUTHORFNAME',
 'AUTHORID',
 'AUTHORVERIFIED',
 'AUTHORDESCRIPTION',
 'AUTHORFOLLOWERS',
 'AUTHORTWEETID',
 'AUTHORTWEETCONTENT',
 'AUTHORTWEETTIMESTAMP']
dfsimple.drop(columns=vartoremove,inplace=True)
dfsimple.dropna(inplace=True)

print("Build dfsimple and dfrt dataframes")


# # Table Influenceur

# In[9]:


keepvar = ["AUTHORID","AUTHORNAME","AUTHORFNAME","AUTHORDESCRIPTION","AUTHORFOLLOWERS"]
TableInf = dfrt.copy()
TableInf = TableInf[keepvar]
TableInf = TableInf.groupby("AUTHORID").first().reset_index()
print("Build batch influenceurs")


# # Table RT

# In[10]:


keepvar = ["TWEETID","AUTHORTWEETID","USERID","AUTHORID","TWEETUNIXEPOCH"]
TableRT = dfrt.copy()
TableRT = TableRT[keepvar]
TableRT["status"] = "ko"
print("Build batch RT")


# # Table Tweet Repris

# In[11]:


keepvar = ["AUTHORID","AUTHORTWEETID","AUTHORTWEETCONTENT","AUTHORTWEETUNIXEPOCH"]
TableFamousTweet = dfrt.copy()
TableFamousTweet = TableFamousTweet[keepvar]
TableFamousTweet.drop_duplicates(inplace = True)
TableFamousTweet["status"] = "ko"
print("Build batch tweet famous")


# # Sauvegarde

# In[12]:


PickleDump(os.path.join(Root,FolderProject,"BatchFamousTweet.pkl"),TableFamousTweet)
PickleDump(os.path.join(Root,FolderProject,"BatchRT.pkl"),TableRT)
PickleDump(os.path.join(Root,FolderProject,"BatchInf.pkl"),TableInf)
print("Save batchs")


# # Logs

# In[13]:


CurrentTimeString = GetCurrentTime()
DateDebut = str(pd.to_datetime(dfrt.TWEETUNIXEPOCH.min(), unit='s'))
DateFin = str(pd.to_datetime(dfrt.TWEETUNIXEPOCH.max(), unit='s'))
SecondsElapsed = dfrt.TWEETUNIXEPOCH.max() - dfrt.TWEETUNIXEPOCH.min()
DelayFormatted = SecondsToDelayFormat(SecondsElapsed)

nRetweets = len(TableRT)
nInfluenceurs = len(TableInf)
nFamousTweets = len(TableFamousTweet)
nUsers = len(np.unique(TableRT.USERID))

BatchLogs = {"BatchDate" : CurrentTimeString,
"StartingDate" : DateDebut,
"EndingDate" : DateFin,
"Delay" : DelayFormatted,
"nRT" : nRetweets,
"nInf" : nInfluenceurs,
"nFam" : nFamousTweets,
"nUse" : nUsers}

filename = os.path.join(Root,FolderProject,"Batch.log")
AppendStringToFile(filename,BatchLogs)

print("Write logs")

