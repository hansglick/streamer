#!/usr/bin/env python
# coding: utf-8

# In[671]:


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


# In[672]:


def LoadJsonFile(filename): 
    with open(filename, 'r') as f:
        DicConfig = json.load(f)
    return DicConfig


def GlobalDicDeplier(OneDic):
    for k,v in OneDic.items():
        exec('globals()[k] = v')
    return None


# In[673]:


print("Chargement du fichier config.json")
print("")
DicConfig = LoadJsonFile(os.path.join(os.getcwd(),"config.json"))
GlobalDicDeplier(DicConfig)
sys.path.append(Root)
from fun import *


# In[675]:


# LOAD DATA
print("Load data")
print("")

path = os.path.join(Root,FolderProject,"RefFam.pkl")
RefFam = LoadPickleOrInit(path)

path = os.path.join(Root,FolderProject,"RefRT.pkl")
RefRT = LoadPickleOrInit(path)

path = os.path.join(Root,FolderProject,"RefInf.pkl")
RefInf = LoadPickleOrInit(path)

path = os.path.join(Root,FolderProject,"tmdic.pkl")
tmdic = LoadPickleOrInit(path,typeobj="dic")


path = os.path.join(Root,FolderProject,"FinalFam.pkl")
FinalFam = LoadPickleOrInit(path)

path = os.path.join(Root,FolderProject,"FinalRT.pkl")
FinalRT = LoadPickleOrInit(path)

path = os.path.join(Root,FolderProject,"FinalInf.pkl")
FinalInf = LoadPickleOrInit(path)


# In[676]:


print("Quel est le prochain Time Mark?")
getnexttm(RefRT,tmdic,StepSize,WindowSize,verbose = True)
print("")


# # Récupérer le next batch de rt et la maj des retweets dataframe

#  * **rtdf** : dataframe de reference des retweets téléchargés. Le fichier est alimenté régulièrement.
#  * **pbest** : proportion de volume de retweets à garder
#  * **tmdic** : dictionnaire dont les clés sont des timemarks, les valeurs peuvent être les bornes inférieurs et supérieurs
#  * **rtdf_period** : dataframe de reference des retweets téléchargés uniquement sur une période
#  * **stepsize** : taille du step de la fenetre glissante en secondes
#  * **windowsize** : taille de la fenêtre glissante en secondes
#  * **tm** : une timemark en secondes

# # Run !

# In[678]:


print("Analytics commence ...")


tocontinue = True
compteur = 0
while(tocontinue):
    bestrtdf,bestfamdf,bestinfdf,informations,nexttm = getbestrtbashdic(RefRT,
                                                                        tmdic,
                                                                        StepSize,
                                                                        WindowSize,
                                                                        TopTweetsProportion,
                                                                        RefFam,
                                                                        RefInf)
    
    if nexttm is not None:
        compteur = compteur + 1
        tmdic[nexttm] = informations
        FinalInf = pd.concat((FinalInf,bestinfdf),axis = 0, sort = True)
        FinalFam = pd.concat((FinalFam,bestfamdf),axis = 0, sort = True)
        FinalRT = pd.concat((FinalRT,bestrtdf),axis = 0, sort = True)

        FinalRT = FinalRT.drop_duplicates()
        FinalFam = FinalFam.drop_duplicates(subset=["AUTHORTWEETID"])
        FinalInf = FinalInf.drop_duplicates(subset=["AUTHORID"])

        FinalInf.reset_index(drop=True,inplace=True)
        FinalFam.reset_index(drop=True,inplace=True)
        FinalRT.reset_index(drop=True,inplace=True)
        
    else:
        tocontinue = False
        
print("Nombre de batch rajoutés : ", compteur)
print("")


# In[680]:


if compteur > 0 :
    PickleDump(os.path.join(Root,FolderProject,"tmdic.pkl"),tmdic)
    PickleDump(os.path.join(Root,FolderProject,"FinalInf.pkl"),FinalInf)
    PickleDump(os.path.join(Root,FolderProject,"FinalFam.pkl"),FinalFam)
    PickleDump(os.path.join(Root,FolderProject,"FinalRT.pkl"),FinalRT)


# In[ ]:





# In[ ]:





# In[681]:


tweetsrt = pd.Series(FinalRT.AUTHORTWEETID.unique()).to_frame(name="AUTHORTWEETID")
solution = tweetsrt.merge(FinalFam,on="AUTHORTWEETID")


# In[683]:


print("Est-ce que tout est ok?")
print(tweetsrt.shape[0] == solution.shape[0])
print("taille : ", len(tweetsrt))
print("")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[685]:


threshold = retrievethreshold(tmdic,WindowSize,RefRT)


# In[686]:


print("Valeur du threshold?")
print(threshold)
print("")


# In[687]:


if threshold is not None : 
    RemoveRT,KeepRT = extractkeepremove(RefRT,threshold)
    PickleDump(os.path.join(Root,FolderProject,"RefRT.pkl"),KeepRT)


# In[688]:


if threshold is not None:
    fil = RefRT.AUTHORTWEETID>=threshold
    tempdf = pd.Series(RefRT.AUTHORTWEETID[fil].unique()).to_frame(name="AUTHORTWEETID")
    toadd = tempdf.merge(RefFam,on="AUTHORTWEETID")
    NewRefFam = pd.concat((toadd,FinalFam),axis=0,sort=True).drop_duplicates(subset="AUTHORTWEETID")


# In[689]:


if threshold is not None:
    PickleDump(os.path.join(Root,FolderProject,"RefFam.pkl"),NewRefFam)


# In[ ]:





# In[690]:


print("Les 5 dernieres time marks : ")
x = list(tmdic.keys())
print([pd.to_datetime(item,unit="s") for item in x[::-1][:5]])
print("")

