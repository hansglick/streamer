#!/usr/bin/env python
# coding: utf-8

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
from scipy import stats
from IPython.display import clear_output, display
import subprocess
from datetime import datetime
pd.options.display.float_format = '{:.0f}'.format


# In[2]:


import pandas as pd
import numpy as np
from spacy.lang.fr.stop_words import STOP_WORDS as fr_stop
french_stopwords = list(fr_stop)
from scipy import stats
import math
import string


# In[3]:


a = {"déjà",
"avez",
"faut",
"êtes",
"faire"}
fr_stop = fr_stop.union(a)


# In[4]:


def LoadJsonFile(filename): 
    with open(filename, 'r') as f:
        DicConfig = json.load(f)
    return DicConfig


def GlobalDicDeplier(OneDic):
    for k,v in OneDic.items():
        exec('globals()[k] = v')
    return None


# In[5]:


print("Chargement du fichier config")
DicConfig = LoadJsonFile(os.path.join(os.getcwd(),"config.json"))
GlobalDicDeplier(DicConfig)
sys.path.append(Root)
from fun import *
print("")


# In[6]:


print("Chargement des donnés")
# LOAD DATA
path = os.path.join(Root,FolderProject,"FinalFam.pkl")
FinalFam = LoadPickleOrInit(path)
print("")


# In[7]:


print("Construction de tweets analyse dataframe")
TweetsToAnalyse = FinalFam.copy()[["AUTHORTWEETID","AUTHORTWEETCONTENT","AUTHORTWEETUNIXEPOCH"]].rename(columns = {"AUTHORTWEETID":"TWEETID","AUTHORTWEETCONTENT":"TWEETCONTENT"})
print("")


# In[ ]:





# In[ ]:





# In[ ]:





# In[8]:


def ExploreTweetTreatment(mytweetid,reffam,tfidf):
    content = reffam.AUTHORTWEETCONTENT[reffam.AUTHORTWEETID==mytweetid].iloc[0]
    tokens = Tokenizer(content)
    details = tfidf[tfidf.tweetid==mytweetid][["word","idf","Rank","f","w"]].    to_dict(orient="records")
    return content,tokens,details


# In[9]:


class Corpus():
    
    # Initialisation de l'objet
    def __init__(self,TooFrequentThreshold,TooInfrequentThreshold):
        
        self.WORDS2TWEETS = pd.DataFrame()
        self.DOCSTOUCHED = pd.DataFrame()
        self.COMPTEUR = {}
        
        self.BATCH_WORDS2TWEETS = pd.DataFrame()
        self.BATCH_DOCSTOUCHED = pd.DataFrame()
        self.BATCH_COMPTEUR = {}
        self.BATCH_V = []
        
        self.DocsRepresentation = pd.DataFrame()
        self.TooFrequentThreshold = TooFrequentThreshold
        self.TooInfrequentThreshold = TooInfrequentThreshold
        self.LastDate = 0
     
    
    def LoadData(self):
        
        self.WORDS2TWEETS = LoadPickleOrInit(os.path.join(Root,FolderProject,"tfidf_words2tweets.pkl"))
        self.DOCSTOUCHED = LoadPickleOrInit(os.path.join(Root,FolderProject,"tfidf_docstouched.pkl"))
        self.COMPTEUR = LoadPickleOrInit(os.path.join(Root,FolderProject,"tfidf_compteur.pkl"),typeobj="dic")
        self.DocsRepresentation = LoadPickleOrInit(os.path.join(Root,FolderProject,"tfidf_DocsRepresentation.pkl"))
        self.LastDate = LoadPickleOrInit(os.path.join(Root,FolderProject,"tfidf_lastdate.pkl"),typeobj="0")
        
        if len(self.DOCSTOUCHED)==0:
            print("No data!")

        return None
    
    
    def SaveOnDisk(self):
        
        PickleDump(os.path.join(Root,FolderProject,"tfidf_compteur.pkl"),self.COMPTEUR)
        PickleDump(os.path.join(Root,FolderProject,"tfidf_docstouched.pkl"),self.DOCSTOUCHED)
        PickleDump(os.path.join(Root,FolderProject,"tfidf_words2tweets.pkl"),self.WORDS2TWEETS)
        PickleDump(os.path.join(Root,FolderProject,"tfidf_DocsRepresentation.pkl"),self.DocsRepresentation)
        PickleDump(os.path.join(Root,FolderProject,"tfidf_lastdate.pkl"),self.LastDate)
        
        return None
    
    
    # EVALUATION de la variable globale |D|
    # i.e. le nombre de documents, i.e. les tweets
    def EvaluateD(self):
        if len(self.WORDS2TWEETS)>0:
            res = len(np.unique(self.WORDS2TWEETS.tweetid))
        else:
            res = 0
        return res
    
    # CREATION de la liste de batch
    def ProcessCorpus(self,TweetsDataFrame):
        
        tweetsdf = TweetsDataFrame.copy()
        
        if len(self.WORDS2TWEETS)>0:
            a = self.WORDS2TWEETS.tweetid.unique()
            b = tweetsdf.TWEETID
            fil = ~b.isin(a)
            tweetsdf = tweetsdf[fil]
            
        tweetsdf = tweetsdf.copy()[tweetsdf.AUTHORTWEETUNIXEPOCH>self.LastDate]
        
        
        print("Nombre de retweets à traier : ",len(tweetsdf))
        nbtweets = len(tweetsdf.TWEETID.unique())
        print("Nombre de tweets uniques : ",nbtweets)
        nsplit = int(nbtweets/10000)+1
        ListOfTweetsDF = np.array_split(tweetsdf, nsplit)
        
        if nbtweets>1000:
            print("Le traitement commence")
            self.WaitingCorpusList = ListOfTweetsDF
            self.LastDate = self.WaitingCorpusList[-1].AUTHORTWEETUNIXEPOCH.max()
        else:
            print("Pas assez de tweets à traiter, réessayer un autre moment")
            self.WaitingCorpusList = []
            
            
        
        return None
        
        
    # TRAITEMENT DE LA LISTE DE BATCH
    def ComputeCorpus(self):
        for iddf,df in enumerate(self.WaitingCorpusList):
            print(str(iddf+1),"/",str(len(self.WaitingCorpusList)))
            self.AddCorpus(df)
            self.BuildDocsRepresentation()
        return None
            
            
    # TRAITEMENT D'UN BATCH
    def AddCorpus(self,tweetsdf):     
        
        if len(tweetsdf)>0:
            
            # Build les batch df
            tweetsdf.TWEETCONTENT = tweetsdf.TWEETCONTENT.map(lambda a : Tokenizer(a))
            Words2TweetsDF = DeplyrDF(tweetsdf)
            CompteurDic = BooleanCorpusCompteur(tweetsdf.TWEETCONTENT.tolist())
            V = list(CompteurDic.keys())
            DocsTouched = BuildDocumentsTouched(V,tweetsdf.TWEETCONTENT.tolist())
            DocsTouched = pd.DataFrame(data=DocsTouched,index=[0]).T.reset_index().rename(columns = {"index":"word",0:"f"})
            
            # Remove les mots trop peu fréquents du batch
            TooFrequentRaw = int(round(self.TooFrequentThreshold * len(Words2TweetsDF.tweetid.unique())))
            TooInfrequentRaw = int(round(self.TooInfrequentThreshold * len(Words2TweetsDF.tweetid.unique())))
            self.TooInfrequentRaw = TooInfrequentRaw
            self.TooFrequentRaw = TooFrequentRaw
            self.Toovalue = len(Words2TweetsDF.tweetid.unique())
            DocsTouched = DocsTouched.copy()[(DocsTouched.f>TooInfrequentRaw) & (DocsTouched.f<TooFrequentRaw)]
            Words2TweetsDF = Words2TweetsDF.merge(DocsTouched,on="word")
            Words2TweetsDF = Words2TweetsDF[["word","tweetid"]]
            V = DocsTouched.word.tolist()
            
            # Assignation des batch df
            self.BATCH_WORDS2TWEETS = Words2TweetsDF
            self.BATCH_DOCSTOUCHED = DocsTouched
            self.BATCH_COMPTEUR = CompteurDic
            self.BATCH_V = V

            
                                                                 
                                                                 
            #  * * * UPDATING PART * * * 
            #  * * * UPDATING PART * * *
            #  * * * UPDATING PART * * *
            
            
            
            
            # maj du dic
            self.COMPTEUR = UpdateCompteurDic(self.COMPTEUR,self.BATCH_COMPTEUR)
            
            # maj du words2tweets
            self.WORDS2TWEETS = pd.concat((self.WORDS2TWEETS,self.BATCH_WORDS2TWEETS),axis=0)
            
            # maj du docstouched
            if len(self.DOCSTOUCHED)>0:
                fil = self.DOCSTOUCHED.word.isin(pd.Series(V))
                nepastoucher = self.DOCSTOUCHED[~fil]
                amodifier = self.DOCSTOUCHED[fil]
                amodifier = pd.concat((amodifier,self.BATCH_DOCSTOUCHED),axis = 0)
                amodifier = amodifier.groupby("word")["f"].sum().reset_index()
                self.DOCSTOUCHED = pd.concat((amodifier,nepastoucher),axis = 0)
            else:
                self.DOCSTOUCHED = self.BATCH_DOCSTOUCHED
                

            
        # si tweetsdf est vide alors les batch sont vides
        else:
            self.BATCH_WORDS2TWEETS = pd.DataFrame()
            self.BATCH_DOCSTOUCHED = pd.DataFrame()
            self.BATCH_COMPTEUR = {}
            self.BATCH_V = []
        
        return None
    
    

    

    
    
    
    def BuildDocsRepresentation(self):
        
        # Si y'a rien dans le batch alors on s'arrête là
        if len(self.BATCH_V)==0:
            return None
        
        # Si y'a déjà des idf scores de calculé on doit la mettre à jour
        # Le keepdf, contient les mots non présents dans le batch
        if len(self.DocsRepresentation)>0:
            fil = self.DocsRepresentation.word.isin(pd.Series(self.BATCH_V))
            DocsRepresentationKeep = self.DocsRepresentation.copy()[~fil]
            DocsRepresentationKeep["D"] = self.EvaluateD()
        else:
            DocsRepresentationKeep = pd.DataFrame()
        
        # df intermediaires
        fil = self.DOCSTOUCHED.word.isin(pd.Series(self.BATCH_V))
        temptouched = self.DOCSTOUCHED.copy()[fil]
        fil = self.WORDS2TWEETS.word.isin(pd.Series(self.BATCH_V))
        tempwords2 = self.WORDS2TWEETS[fil]        
        toadd = temptouched.merge(tempwords2,on="word")
        toadd["D"] = self.EvaluateD()   
        
        # concaténation des scores des anciens mots et des nouveaux mots
        solution = pd.concat((toadd,DocsRepresentationKeep),axis = 0, sort = True)
        solution["idf"] = (solution.D / solution.f).map(lambda a : math.log(a))
        solution = solution
        solution.reset_index(drop = True,inplace = True)
        solution.sort_values(by = ["tweetid","idf"],ascending=False,inplace=True)
        solution.drop_duplicates(inplace = True)
        self.DocsRepresentation = solution
        
        return None
        
        
        
        
        
        
        


# In[10]:


def Tokenizer(randomstring,french_stopwords=french_stopwords):
    translator = str.maketrans(string.punctuation,' '*32)
    randomstring = randomstring.translate(translator)
    randomstring = randomstring.replace("’"," ")
    randomstring = randomstring.replace("`"," ")
    randomstring = randomstring.replace("'"," ")
    randomstring = randomstring.replace("“"," ")
    randomstring = randomstring.replace("”"," ")
    randomstring = randomstring.replace("…"," ")
    randomstring = randomstring.lower()
    randomstring = " ".join(randomstring.split())
    words = randomstring.split(" ")
    words = [item for item in words if item not in french_stopwords]
    words = [item for item in words if len(item)>2]
    return words


# In[11]:


def SentenceCompteur(compteur,words,weight=1):
    for w in words:
        compteur[w] = compteur.get(w,0) + weight
    return compteur

def BooleanCorpusCompteur(Corpus):
    compteur = {}
    for doc in Corpus:
        compteur = SentenceCompteur(compteur,doc)
    return compteur


# In[12]:


def BuildDocumentsTouched(V,CleanedCorpus):
    
    DocumentsTouched = {}
    for w in V:
        for d in CleanedCorpus:
            if w in d:
                DocumentsTouched[w] = DocumentsTouched.get(w,0) + 1        
    
    return DocumentsTouched


# In[13]:


def UpdateCompteurDic(OriginalDic,AddDic):
    for k,v in AddDic.items():
        OriginalDic[k] = OriginalDic.get(k,0) + v
    return OriginalDic


# In[14]:


def DeplyrDF(TweetsDataFrame):

    L = []
    for i,row in TweetsDataFrame.iterrows():
        tweetid = row["TWEETID"]
        tweetcontent = row["TWEETCONTENT"]
        tweetcontent = pd.Series(tweetcontent)
        tempdf = tweetcontent.to_frame(name = "word")
        tempdf["tweetid"] = tweetid
        L.append(tempdf)

    res = pd.concat(L,axis=0)
    res.reset_index(drop=True,inplace=True)

    return res


# In[15]:


def SplitTweetsToAnalyse(TweetsToAnalyse,RemoveWordsPeriod):
    TweetsToAnalyse["TimeElapsed"] = (TweetsToAnalyse.AUTHORTWEETUNIXEPOCH.max() - TweetsToAnalyse.AUTHORTWEETUNIXEPOCH) / RemoveWordsPeriod
    TweetsToAnalyse["TimeElapsed"] = TweetsToAnalyse["TimeElapsed"].astype(int)
    idgroup = TweetsToAnalyse.TimeElapsed.unique()
    idgroup.sort()
    L = []
    for idg in idgroup[:-1]:
        tempdf = TweetsToAnalyse.copy()[TweetsToAnalyse.TimeElapsed==idg]
        L.append(tempdf)
    return L


# In[ ]:





# In[ ]:





# In[ ]:





# In[16]:


print("Analytics commence ...")
corpus = Corpus(TooFrequentThreshold,TooInfrequentThreshold)
corpus.LoadData()
corpus.ProcessCorpus(TweetsToAnalyse)
corpus.ComputeCorpus()
corpus.SaveOnDisk()
print("")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




