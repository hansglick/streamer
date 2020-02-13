#!/usr/bin/env python
# coding: utf-8

# In[140]:


import pandas as pd
import json
import pickle
import os
import itertools
from ast import literal_eval
import numpy as np
import sys
from datetime import datetime
from scipy import stats
from IPython.display import clear_output, display
import subprocess
from datetime import datetime
pd.options.display.float_format = '{:.0f}'.format


# In[141]:


def LoadJsonFile(filename): 
    with open(filename, 'r') as f:
        DicConfig = json.load(f)
    return DicConfig


def GlobalDicDeplier(OneDic):
    for k,v in OneDic.items():
        exec('globals()[k] = v')
    return None


# In[142]:


print("Chargement du fichier config")
DicConfig = LoadJsonFile(os.path.join(os.getcwd(),"config.json"))
GlobalDicDeplier(DicConfig)
sys.path.append(Root)
from fun import *
print("")


# In[143]:


print("Chargement des données")

path = os.path.join(Root,FolderProject,"FinalFam.pkl")
FinalFam = LoadPickleOrInit(path)

path = os.path.join(Root,FolderProject,"FinalRT.pkl")
FinalRT = LoadPickleOrInit(path)

path = os.path.join(Root,FolderProject,"FinalInf.pkl")
FinalInf = LoadPickleOrInit(path)

path = os.path.join(Root,FolderProject,"gdic.pkl")
gdic = LoadPickleOrInit(path,typeobj="dic")

print("")


# In[144]:


def getnexttm(rtdf,tmdic,stepsize,graphwin,tfidfwin,verbose = False):
    """
    input : un dictionnaire de timemark
    input : le dataframe de retweets reference
    input : le stepsize
    output : soit un timemark s'il existe, soit un None Type
    """
    
    # Le window size
    if graphwin>=tfidfwin:
        windowsize = graphwin
    else:
        windowsize = tfidfwin
    
    # Quel est le timestamp du dernier retweet?
    lastretweetts = rtdf.TWEETUNIXEPOCH.max()
    firstretweetts = rtdf.TWEETUNIXEPOCH.min()

    # Quel est le dernier timemark?
    if len(tmdic)>0:
        tmlist = list(tmdic.keys())
        lasttm = max(tmlist) + stepsize
    else:
        lasttm = firstretweetts + windowsize
        
    if verbose:
        print("Date du du potentiel next Time Mark : ",pd.to_datetime(lasttm,unit="s"))
        print("Date 1er retweet : ",pd.to_datetime(firstretweetts,unit="s"))
        print("Date Dernier retweet : ",pd.to_datetime(lastretweetts,unit="s"))
        print("")
    
    # ESPACE AVANT
    forwardcdt = lastretweetts>lasttm
    
    if forwardcdt:
        return lasttm
    else:
        return None


# In[145]:


def ExtractRawDF(nexttm,FinalRT,GraphWindowSize,GraphTfidfSize):

    if nexttm is not None:
        bornesup = nexttm
        borneinfGraph = nexttm - GraphWindowSize
        borneinfTfidf = nexttm - GraphTfidfSize

        fil = (FinalRT.TWEETUNIXEPOCH>=borneinfGraph) & (FinalRT.TWEETUNIXEPOCH<bornesup)
        rawgraphdf = FinalRT[fil]

        fil = (FinalRT.TWEETUNIXEPOCH>=borneinfTfidf) & (FinalRT.TWEETUNIXEPOCH<bornesup)
        rawtfidfdf = FinalRT[fil]
        


    else:
        rawgraphdf = None
        rawtfidfdf = None

    return rawgraphdf,rawtfidfdf


# In[146]:


def ExtractGraphDF(rawgraphdf,rawtfidfdf,FinalFam):

    if rawgraphdf is not None:
        rtstatsdf = rawgraphdf.groupby("AUTHORTWEETID").size().reset_index().rename(columns = {0:"f"})
        list_of_authors = rawgraphdf.groupby('USERID')['AUTHORID'].apply(list)
        LinksDic = GetLinksFromPeriod(list_of_authors)
    else:
        LinksDic = None
        rtstatsdf = None

    if rawtfidfdf is not None:
        tweetsdf = pd.Series(rawtfidfdf.AUTHORTWEETID.unique()).to_frame(name="AUTHORTWEETID").merge(FinalFam,on="AUTHORTWEETID")
        tweetsdf.reset_index(drop=True,inplace=True)
    else:
        tweetsdf = None

    return LinksDic,tweetsdf,rtstatsdf


# In[147]:


def RunOnePassGraph(FinalRT,gdic,GraphStepSize,GraphWindowSize,GraphTfidfSize,FinalFam):

    nexttm = getnexttm(FinalRT,gdic,GraphStepSize,GraphWindowSize,GraphTfidfSize)
    rawgraphdf,rawtfidfdf = ExtractRawDF(nexttm,FinalRT,GraphWindowSize,GraphTfidfSize)
    LinksDic,tweetsdf,rtstatsdf = ExtractGraphDF(rawgraphdf,rawtfidfdf,FinalFam)

    cdta = nexttm is not None
    cdtb = LinksDic is not None
    cdtc = tweetsdf is not None
    cdtd = rtstatsdf is not None

    if cdta and cdtb and cdtc and cdtd:
        informations = tweetsdf.merge(rtstatsdf,on="AUTHORTWEETID")
        gdic[nexttm] = {"links":LinksDic,
                        "informations" : informations[["AUTHORTWEETID","AUTHORID","AUTHORTWEETUNIXEPOCH","f"]],
                        "tmstr":pd.to_datetime(nexttm,unit="s"),
                        "tm":nexttm}
        tocontinue = True
    else:
        tocontinue = False

    return gdic,tocontinue


# In[148]:


print("Analytics commence ...")
tocontinue = True
compteur = 0
while tocontinue:
    gdic,tocontinue = RunOnePassGraph(FinalRT,gdic,GraphStepSize,GraphWindowSize,GraphTfidfSize,FinalFam)
    if tocontinue:
        compteur = compteur + 1
        print(compteur)
print("Nombre graph rajoutés : ",compteur)
print("")


# In[149]:


print("Enregistrement du gdic")
PickleDump(os.path.join(Root,FolderProject,"gdic.pkl"),gdic)
print("")


# In[150]:


print("Nettoyage du FinalRT")
### Nettoie le finalRT s'il est trop vieux
threshold = retrievethreshold(gdic,max(GraphWindowSize,GraphTfidfSize),FinalRT)
RemoveFinalRT,KeepFinalRT = extractkeepremove(FinalRT,threshold)
PickleDump(os.path.join(Root,FolderProject,"FinalRT.pkl"),KeepFinalRT)
print("")


# In[153]:


print("Nombre de graphs enregistrés : ",len(gdic))
print("")

