#!/usr/bin/env python
# coding: utf-8

# In[1]:


print("   * * *   Lancement du BatchBest.py   * * *   ")
print("")


# In[2]:


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


# In[3]:


def LoadJsonFile(filename): 
    with open(filename, 'r') as f:
        DicConfig = json.load(f)
    return DicConfig

def GlobalDicDeplier(OneDic):
    for k,v in OneDic.items():
        exec('globals()[k] = v')
    return None


# In[4]:


print("Chargement du fichier config")

DicConfig = LoadJsonFile(os.path.join(os.getcwd(),"config.json"))
GlobalDicDeplier(DicConfig)
sys.path.append(Root)
from fun import *

print("")


# In[5]:


print("Chargement des données")

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

print("")


# In[6]:


print("Calcul des statistiques pour les logs")

FamOriginSize = len(FinalFam)
RTOriginSize = len(FinalRT)
InfOriginSize = len(FinalInf)

print("")


# In[7]:


print("Informations quant aux prochains time mark")
nexttmsaved = getnexttm(RefRT,tmdic,StepSize,WindowSize,verbose = True)
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

# In[8]:


print("Début de l'Analytics ...")

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


# In[9]:


if compteur > 0 :
    print("Sauvegarde du dictionnaire tmdic et des dataframes Finalx")
    
    PickleDump(os.path.join(Root,FolderProject,"tmdic.pkl"),tmdic)
    PickleDump(os.path.join(Root,FolderProject,"FinalInf.pkl"),FinalInf)
    PickleDump(os.path.join(Root,FolderProject,"FinalFam.pkl"),FinalFam)
    PickleDump(os.path.join(Root,FolderProject,"FinalRT.pkl"),FinalRT)
    
    print("")


# In[10]:


print("Vérification cohérence")

tweetsrt = pd.Series(FinalRT.AUTHORTWEETID.unique()).to_frame(name="AUTHORTWEETID")
solution = tweetsrt.merge(FinalFam,on="AUTHORTWEETID")

print("Fichier cohérent",tweetsrt.shape[0] == solution.shape[0])
print("taille : ", len(tweetsrt))
print("")


# In[11]:


print("Valeur du seuil en dessous duquel on peut supprimer une partie de la table RefRT:")

threshold = retrievethreshold(tmdic,WindowSize,RefRT)

print(str(pd.to_datetime(threshold,unit="s")))
print("")

if threshold is not None : 
    RemoveRT,KeepRT = extractkeepremove(RefRT,threshold)
    PickleDump(os.path.join(Root,FolderProject,"RefRT.pkl"),KeepRT)
    print("Modification de RefRT, table de référence des retweets")
    print("")
else:
    KeepRT = pd.DataFrame()
    
if threshold is not None:
    fil = RefRT.AUTHORTWEETID>=threshold
    tempdf = pd.Series(RefRT.AUTHORTWEETID[fil].unique()).to_frame(name="AUTHORTWEETID")
    toadd = tempdf.merge(RefFam,on="AUTHORTWEETID")
    NewRefFam = pd.concat((toadd,FinalFam),axis=0,sort=True).drop_duplicates(subset="AUTHORTWEETID")
    
if threshold is not None:
    PickleDump(os.path.join(Root,FolderProject,"RefFam.pkl"),NewRefFam)
    print("Modification du fichier RefFam, table de référence des tweets repris")
    print("")
else:
    NewRefFam = pd.DataFrame()


# In[12]:


print("Ecriture des logs")

mylog = {
"Date":GetCurrentTime(),
"Nombre de batchs rajoutés" : compteur,
"Taille du dictionnaire tmdic" : len(tmdic),
"Le premier time mark" : nexttmsaved,
"Nombre de lignes rajoutées dans la table RefFam":len(RefFam) - FamOriginSize,
"Nombre de lignes rajoutées dans la table RefRT":len(RefRT) - RTOriginSize,
"Nombre lignes rajoutées dans la table RefInf":len(RefInf) - InfOriginSize,
"Suppression des lignes dans la table RefRT" : len(RefRT) - len(KeepRT),
"Suppression des lignes dans la table RefFam":len(RefFam)  - len(NewRefFam)
}

filename = os.path.join(Root,FolderProject,"Best.log")
AppendStringToFile(filename,mylog)

print("")


# In[ ]:




