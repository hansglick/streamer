#!/usr/bin/env python
# coding: utf-8

# In[76]:


print("   * * *   Lancement de Prerequisites.py   * * *   ")
print("")


# # dans le gdic on doit absolument ajouter les bornes aux deux dataframes construits

# In[77]:


import pandas as pd
import matplotlib.pyplot as plt
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


# In[78]:


import pandas as pd
import numpy as np
from spacy.lang.fr.stop_words import STOP_WORDS as fr_stop
french_stopwords = list(fr_stop)
from scipy import stats
import math
import string
from collections import Counter
import igraph as ig


# In[79]:


def LoadJsonFile(filename): 
    with open(filename, 'r') as f:
        DicConfig = json.load(f)
    return DicConfig


def GlobalDicDeplier(OneDic):
    for k,v in OneDic.items():
        exec('globals()[k] = v')
    return None


# In[80]:


print("Chargement du fichier config")

DicConfig = LoadJsonFile(os.path.join(os.getcwd(),"config.json"))
GlobalDicDeplier(DicConfig)
sys.path.append(Root)
from fun import *

print("")


# In[81]:


print("Chargement des donnés")

path = os.path.join(Root,FolderProject,"FinalFam.pkl")
FinalFam = LoadPickleOrInit(path)

path = os.path.join(Root,FolderProject,"FinalInf.pkl")
FinalInf = LoadPickleOrInit(path)

path = os.path.join(Root,FolderProject,"gdic.pkl")
gdic = LoadPickleOrInit(path)

path = os.path.join(Root,FolderProject,"tfidf_DocsRepresentation.pkl")
DocsRep = LoadPickleOrInit(path)
DocsRep.rename(columns = {"f":"ndocs","tweetid":"AUTHORTWEETID"},inplace=True)

path = os.path.join(Root,FolderProject,"tmdic.pkl")
tmdic = LoadPickleOrInit(path)

print("")


# In[82]:


print("Chargement des fonctions")


# In[83]:


def ComputeStatsWord(DocsRep,gtemp):

    statstemp= DocsRep.merge(gtemp,on="AUTHORTWEETID")
    statstemp["power"] = statstemp.idf * statstemp.f
    statswords = statstemp.groupby("word")["power"].sum().reset_index()
    statswords.sort_values(by="power",inplace=True)
    statswords.reset_index(drop=True,inplace=True)
    statswords["powernormalized"] = statswords.power / statswords.power.sum()

    return statswords


# In[84]:


def LinksFromDicToDF(randomlinksdf):

    L = []
    for k,v in randomlinksdf.items():
        a = k[0]
        b = k[1]
        f = v
        tempdic = {"a":a,"b":b,"f":f}
        L.append(tempdic)

    solution = pd.DataFrame(L)

    return solution


# In[85]:


def BuildAndSavePRQDF(gdic,DocsRep,Root,FolderProject,FolderPRQ):
    
    compteur = 0
    linksrows = 0
    statsrows = 0
    for k,v in gdic.items():
        if "ondisk" not in gdic[k]:
            linkscsv = LinksFromDicToDF(v["links"])
            linkscsv["tmstr"] = str(v["tmstr"])
            linkscsv["start"] = v["GraphStart"]
            linkscsv["end"] = v["tm"]
            path = os.path.join(Root,FolderProject,FolderPRQ,"graph"+str(k)+".csv")
            linkscsv.to_csv(path,index=False)
            linksrows = linksrows + len(linkscsv)

            gtemp = v["informations"]
            statswords = ComputeStatsWord(DocsRep,gtemp)
            statswords["tmstr"] = str(v["tmstr"])
            statswords["start"] = v["TFIDFStart"]
            statswords["end"] = v["tm"]
            path = os.path.join(Root,FolderProject,FolderPRQ,"tfidf"+str(k)+".csv")
            statswords.to_csv(path,index=False)
            statsrows = statsrows + len(statswords)

            gdic[k]["ondisk"] = True
            compteur = compteur + 1
        
    return compteur,linksrows,statsrows


# In[86]:


def ExtractNextBorne(SelectedID,x,verbose=False):

    try:
        InfValue = x[SelectedID][1]
        if verbose:
            print(InfValue)
        
        ChampDesPossibles = np.array([item[0] for item in x[SelectedID + 1 :len(x)]])
        if verbose:
            print(ChampDesPossibles)
        
        Distance = np.abs(InfValue - ChampDesPossibles)
        if verbose:
            print(Distance)
        
        idres = np.argmin(Distance) + SelectedID + 1 
        if verbose:
            print(idres)
        
        res = x[idres]
        if verbose:
            print(res)
        
    except:
        idres = None
        res = None
    
    return idres,res


# In[87]:


def ExtractAllBornes(x):

    """
    Input : liste de tuples représentant les bornes
    Output : liste de tuples représentant les bornes les plus revelantes
    """
    
    x.sort(key = lambda a : a[1], reverse = False)
    SelectedID = 0
    borne = x[SelectedID]
    L = [borne]

    while True:
        idres,res = ExtractNextBorne(SelectedID,x)
        if idres is not None:
            SelectedID = idres
            L.append(res)
        else:
            break

    return L


# In[88]:


def BuildGraphFrom2DF(FinalInf,graphdf):

    """
    Input : un dataframe des nodes
    Input : un dataframe des links
    Output : un objet graph igraph
    """
    
    g = ig.Graph(directed=False)
    nodesids = FinalInf.GRAPHID.values
    g.add_vertices(nodesids)

    g.vs["AUTHORDESCRIPTION"] = FinalInf.AUTHORDESCRIPTION.tolist()
    g.vs["AUTHORFNAME"] = FinalInf.AUTHORFNAME.tolist()
    g.vs["AUTHORID"] = FinalInf.AUTHORID.tolist()
    g.vs["AUTHORNAME"] = FinalInf.AUTHORNAME.tolist()
    g.vs["AUTHORFOLLOWERS"] = FinalInf.AUTHORFOLLOWERS.tolist()

    edgeslist = graphdf[["GRAPHIDA","GRAPHIDB"]].values.tolist()
    g.add_edges(edgeslist)
    g.es["weight"] = graphdf.f.tolist()

    return g


def DeleteSomeNodes(g,AttributeName,ThresholdFollowersInf,ThresholdFollowersSup):

    """
    Input : un objet igraph graph
    Input : borne inferieur et superieur
    Input : Attribute Name
    Output : un objet igraph graph
    """
    
    
    to_delete_ids = [node for node in g.vs if node[AttributeName]<ThresholdFollowersInf]
    g.delete_vertices(to_delete_ids)

    to_delete_ids = [node for node in g.vs if node[AttributeName]>ThresholdFollowersSup]
    g.delete_vertices(to_delete_ids)

    return g


# In[89]:


def SumUpDics(listofdics):
    
    mydics = [Counter(item) for item in listofdics]
    c = Counter()
    
    for d in mydics:
        c.update(d)
    
    return c


# In[90]:


def BuildGlobalLinksDF(gdic):

    BornesList = [(v["GraphStart"],v["tm"]) for k,v in gdic.items()]
    MyBornes = ExtractAllBornes(BornesList)
    TimeMarkList = [item[1] for item in MyBornes]
    ListOfSelectedDataFrames = [gdic[item]["links"] for item in gdic.keys() if item in TimeMarkList]
    GlobalLinksdf = dict(SumUpDics(ListOfSelectedDataFrames))
    GlobalBorne = (MyBornes[0][0],MyBornes[-1][1])
    GlobalLinksdf = LinksFromDicToDF(GlobalLinksdf)
    GlobalLinksdf = GlobalLinksdf.sort_values(by="f",ascending=False).reset_index(drop=True)

    return GlobalBorne,GlobalLinksdf


# In[91]:


print("")


# In[ ]:





# In[ ]:





# In[ ]:





# In[92]:


print("Sauvegarde des fichiers PRQ, input de l'app Shiny")

compteur,linksrows,statsrows = BuildAndSavePRQDF(gdic,DocsRep,Root,FolderProject,FolderPRQ)

path = os.path.join(Root,FolderProject,FolderPRQ,"FinalFam.csv")
FinalFam.to_csv(path,index=False)

path = os.path.join(Root,FolderProject,FolderPRQ,"FinalInf.csv")
FinalInf.to_csv(path,index=False)

PickleDump(os.path.join(Root,FolderProject,"gdic.pkl"),gdic)
print("Nombre de dataframes rajoutés : ",compteur)
print("Nombre de lignes rajoutées, links : ",linksrows)
print("Nombre de lignes rajoutées, statswords : ",statsrows)
print("")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[93]:


GlobalBorne,GlobalLinksdf = BuildGlobalLinksDF(gdic)


# In[94]:


graphdf = GlobalLinksdf.copy()
graphdf.sort_values(by="f",ascending=False,inplace=True)
nodesids = np.unique(np.hstack((graphdf.a.values,graphdf.b.values)))

FinalInf = PickleLoad("FinalInf.pkl")
FinalInf = FinalInf[FinalInf.AUTHORID.isin(nodesids)]
FinalInf = FinalInf.reset_index(drop=True).reset_index()
FinalInf.rename(columns = {"index":"GRAPHID"},inplace=True)

graphdf = graphdf.merge(FinalInf[["GRAPHID","AUTHORID"]],how="left",left_on="a",right_on="AUTHORID").rename(columns={"GRAPHID":"GRAPHIDA"})
graphdf = graphdf.merge(FinalInf[["GRAPHID","AUTHORID"]],how="left",left_on="b",right_on="AUTHORID").rename(columns={"GRAPHID":"GRAPHIDB"})
graphdf.drop(columns=["AUTHORID_x","AUTHORID_y"],inplace = True)


# In[95]:


# Meta Parameters

ThresholdFollowersInf = 40000
ThresholdFollowersSup = 80000
maxComponents = 6
maxWeight = 2


# In[96]:


g = BuildGraphFrom2DF(FinalInf,graphdf)


# In[97]:


g = BuildGraphFrom2DF(FinalInf,graphdf)
todelete = [edge for edge in g.es if edge["weight"]<maxWeight]
g.delete_edges(todelete)
g = DeleteSomeNodes(g,"AUTHORFOLLOWERS",ThresholdFollowersInf,ThresholdFollowersSup)
todelete = [node for node in g.vs if node.degree()==0]
g.delete_vertices(todelete)


# In[98]:


# Filtrer le dataframe, seulement le top5 des components

ComponentsSize = np.array([len(item) for item in g.components()])
ComponentsSize = np.sort(ComponentsSize)[::-1]
size,freq = np.unique(ComponentsSize,return_counts=True)
size = size[::-1]
freq = freq[::-1]
nodesids = [item for item in g.components() if len(item) in list(size[freq.cumsum()<maxComponents])]
nodesids = [item for sublist in nodesids for item in sublist]
todelete = [node for node in g.vs if node.index not in nodesids]
g.delete_vertices(todelete)


# In[99]:


# Clustering

clustering = g.community_multilevel(weights="weight")


# In[100]:


# Construction des nodesdf et linksdf

nodesdf = pd.Series(clustering.membership).to_frame(name="CLUSTERID").reset_index().rename(columns = {"index":"GRAPHINDEX"})
nodesdf["GRAPHID"] = pd.Series(g.vs["name"])
nodesdf["AUTHORID"] = pd.Series(g.vs["AUTHORID"])
nodesdf["AUTHORFNAME"] = pd.Series(g.vs["AUTHORFNAME"])
nodesdf["AUTHORNAME"] = pd.Series(g.vs["AUTHORNAME"])
nodesdf["AUTHORFOLLOWERS"] = pd.Series(g.vs["AUTHORFOLLOWERS"])
nodesdf["PERIODINF"] = GlobalBorne[0]
nodesdf["PERIODSUP"] = GlobalBorne[1]


# In[101]:


data = [(item.source,item.target,item.attributes()["weight"]) for item in g.es]
linksdf = pd.DataFrame(data, columns=['a', 'b', 'weight'])

linksdf = linksdf.merge(nodesdf[["GRAPHINDEX","AUTHORID"]],left_on="a",right_on="GRAPHINDEX",how="left").drop(columns=["GRAPHINDEX"]).rename(columns = {"AUTHORID":"AUTHORIDA"})

linksdf = linksdf.merge(nodesdf[["GRAPHINDEX","AUTHORID"]],left_on="b",right_on="GRAPHINDEX",how="left").drop(columns=["GRAPHINDEX"]).rename(columns = {"AUTHORID":"AUTHORIDB"})

linksdf["PERIODINF"] = GlobalBorne[0]
linksdf["PERIODSUP"] = GlobalBorne[1]


# In[102]:


# Sauvegarde des dataframes sur le disk
nodesdf.to_csv(os.path.join(Root,FolderProject,FolderPRQ,"nodesdf.csv"),index=False)
linksdf.to_csv(os.path.join(Root,FolderProject,FolderPRQ,"linksdf.csv"),index=False)


# In[103]:


print("Ecriture des logs")

files = os.listdir(os.path.join(Root,FolderProject,FolderPRQ))
graphfilesnumber = len([item.startswith("graph") for item in files])
tfidffilesnumber = len([item.startswith("tfidf") for item in files])
nvertex = len(g.vs)
nedges = len(g.es)
nclusters = len(np.unique(np.array(clustering.membership)))
ncompo = len(g.components())

log = {
    "Date":GetCurrentTime(),
    "Nombre de graph dataframes crées" : compteur,
    "Nombre de lignes total rajoutées, linksdf":linksrows,
    "Nombre de lignes total rajoutées, statswords":statsrows,
    "Nombre de tweets, FinalFam":len(FinalFam),
    "Nombre d'auteurs, FinalInf":len(FinalInf),
    "Nombre de dataframes de type graph so far":graphfilesnumber,
    "Nombre de dataframes de type tfidf so far":tfidffilesnumber,
    "Nombre de vertex dans le graph global":nvertex,
    "Nombre de links dans le graph global":nedges,
    "Nombre de communautés dans le graph global":nclusters,
    "Nombre de composantes principales dans le graph global":ncompo
}

filename = os.path.join(Root,FolderProject,"PRQ.log")
AppendStringToFile(filename,log)

print("")

