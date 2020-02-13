import json
import sys
import time
from tweepy import OAuthHandler, Stream
from tweepy.streaming import StreamListener
from tweepy import API
import os
import itertools
from ast import literal_eval
import numpy as np
from datetime import datetime
from IPython.display import clear_output, display
import json
from shutil import copyfile
import pandas as pd
import json
import pickle





def DefineAuthentication(filename):

	auth = []
	f = open(filename, "r")
	for line in f:
	    auth.append(line.strip())
	f.close()

	return auth


class TwitterListener(StreamListener):

    def on_data(self, data):
        ParsingFunction(data)
        return(True)

    def on_error(self, status):
        print(status)



def LauchScrapping(AuthFilename,TrendingTopics,LangList):

	# Authentification du compte Twitter
	auth = DefineAuthentication(AuthFilename)

	# Création du Listener
	Listener = TwitterListener()

	# Création de l'authitificateur
	auth_key = OAuthHandler(auth[0], auth[1])
	auth_key.set_access_token(auth[2], auth[3])

	# Création du LiveStream
	LiveStream = Stream(auth_key, Listener)

	# Création de l'instance API
	api= API(auth_key)

	# Lauch Scrapping
	LiveStream.filter(track=TrendingTopics, languages = LangList)

	return None


def ParsingFunction(StreamedData):

	# Load Json
	tweet = json.loads(StreamedData)
	rtbool = False

	# Storing Data
	StructuredInformations = dict(

		# User Informations
		USERNAME = str(tweet["user"]["screen_name"]),
		USERFNAME = str(tweet["user"]["name"]),
		USERID = str(tweet["user"]["id_str"]),
		USERVERIFIED = str(tweet["user"]["verified"]),
		USERDESCRIPTION = str(tweet["user"]["description"]),
		USERFOLLOWERS = str( tweet["user"]["followers_count"]),

		# Tweet Informations
		TWEETID = str(tweet["id_str"]),
		TWEETCONTENT = str(tweet["text"]),
		TWEETHASHTAGS = tweet["entities"]["hashtags"],
		TWEETTIMESTAMP = str(tweet["created_at"])
		)

	# Storing Retweet informations
	if "retweeted_status" in tweet:
		rtbool = True
		RetweetInformations = dict(

			# Author Informations
			AUTHORNAME = str(tweet["retweeted_status"]["user"]["screen_name"]),
			AUTHORFNAME = str(tweet["retweeted_status"]["user"]["name"]),
			AUTHORID = str(tweet["retweeted_status"]["user"]["id_str"]),
			AUTHORVERIFIED = str(tweet["retweeted_status"]["user"]["verified"]),
			AUTHORDESCRIPTION = str(tweet["retweeted_status"]['user']['description']),
			AUTHORFOLLOWERS = str(tweet["retweeted_status"]['user']['followers_count']),

			# Tweet Author Description
			AUTHORTWEETID = str(tweet["retweeted_status"]["id_str"]),
			AUTHORTWEETCONTENT = str(tweet["retweeted_status"]["text"]),
			AUTHORTWEETTIMESTAMP = str(tweet["retweeted_status"]["created_at"]),

			# Retweet Boolean
			ISRETWEET = rtbool

			)

	else:
		rtbool = False
		RetweetInformations = dict(
			AUTHORNAME ="",
			AUTHORFNAME ="",
			AUTHORID = "",
			AUTHORVERIFIED = "",
			AUTHORDESCRIPTION ="",
			AUTHORFOLLOWERS = "",
			AUTHORTWEETID = "",
			AUTHORTWEETCONTENT = "",
			AUTHORTWEETTIMESTAMP = "",
			ISRETWEET = rtbool
			)


	# Concatening Stuff and Printing
	FinalDic = {**StructuredInformations, **RetweetInformations}
	print(FinalDic)


	return None


def ListenningMessages(AuthFilename,TrendingTopics,LangList):
	while True:
		
		try:
			LauchScrapping(AuthFilename,TrendingTopics,LangList)
		
		except KeyboardInterrupt as e:
			break
		
		except:
			continue
	return None
























def CreateFolder(Root,FolderProject):

    AbsoluteFolderPath = os.path.join(Root,FolderProject)
    AbsConfigPath = os.path.join(AbsoluteFolderPath,"config.json")
    
    if not os.path.exists(AbsoluteFolderPath):
        os.makedirs(AbsoluteFolderPath)
    else:
        with open(AbsConfigPath, 'r') as f:
            DicConfig = json.load(f)
        print(DicConfig['TrendingTopics'])

    return None


def SaveConfig(DicConfig,Root,FolderProject):
    
    AbsoluteFolderPath = os.path.join(Root,FolderProject)
    AbsConfigPath = os.path.join(AbsoluteFolderPath,"config.json")
    
    with open(AbsConfigPath, 'w') as f:
        json.dump(DicConfig, f)
    
    return None


def WriteStreamPythonString(Root,FolderProject):
    
    cmd1 = "Root = '"+ Root+"'"
    cmd2 = "FolderProject ='" + FolderProject + "'"

    a = """
    import json
    import os
    """
    b = "\n"+cmd1+"\n"+cmd2+"\n"
    c = """
    AbsConfigPath = os.path.join(Root,FolderProject,'config.json')

    with open(AbsConfigPath, 'r') as f:
        DicConfig = json.load(f)

    for k,v in DicConfig.items():
        exec(k + '=' + 'v')

    import sys
    import time
    from tweepy import OAuthHandler, Stream
    from tweepy.streaming import StreamListener
    from tweepy import API
    sys.path.append(Root)
    from fun import *

    AuthFilename = os.path.join(Root,AuthFilename)

    ListenningMessages(AuthFilename,TrendingTopics,LangList)
    """
    PythonScript = a.replace("\n    ","\n") + b.replace("\n    ","\n") + c.replace("\n    ","\n")

    return PythonScript


def WriteStringToScript(StringScript,Path):

    with open(Path, 'w') as out:
        out.write(StringScript)

    return None


def WriteStreamBashString(PythonRoot,TweetsFilename):
    
    a = PythonRoot
    b = " RunTracking.py 1>>"
    c = TweetsFilename
    BashScript = a + b + c

    return BashScript



def CopyFilesToDirectory(OriginalFilesToCopy,SourceFolderPath,TargetFolderPath):
    for af in OriginalFilesToCopy:
        AbsSourcePath = os.path.join(SourceFolderPath,af)
        AbsDestinationPath = os.path.join(TargetFolderPath,af)
        copyfile(AbsSourcePath, AbsDestinationPath)
    return None















def LoadJsonFile(filename): 
    with open(filename, 'r') as f:
        DicConfig = json.load(f)
    return DicConfig


def GlobalDicDeplier(OneDic):
    for k,v in OneDic.items():
        exec('globals()[k] = v')
    return None


def ReadTweetsToList(Root,FolderProject,TweetsFilename,sizetweet=20):

    errors = 0
    errors420 = 0
    filename = os.path.join(Root,FolderProject,TweetsFilename)
    tweetslist = []

    with open(filename) as fp:
        for line in fp:
            try:
                jsontweet = literal_eval(line)
                if jsontweet == 420:
                    errors420 = errors420 + 1
                if type(jsontweet) is dict and len(jsontweet)==sizetweet:
                    tweetslist.append(jsontweet)
                else:
                    errors = errors +1
            except:
                errors = errors + 1

    if errors420>0:
        print("Error 420 : ",str(errors420))
    else:
        print("Aucune Error 420")

    return tweetslist,errors



def PickleDump(filename,objitem):

    outfile = open(filename,'wb')
    pickle.dump(objitem,outfile)
    outfile.close()

    return None




def LoadPickleOrInit(path,typeobj="df"):

    DoesExist = os.path.isfile(path)
    if DoesExist:
        SomeObj = PickleLoad(path)
    else:
        if typeobj=="df":
            SomeObj = pd.DataFrame()
        if typeobj=="dic":
            SomeObj = {}
        if typeobj=="0":
            SomeObj = 0

    return SomeObj


def PickleLoad(path):
    
    f = open(path, 'rb')
    someobj = pickle.load(f)     
    f.close()
    
    return someobj








def ConvertAndRunScript(JupyterRoot,PythonRoot,Root,FolderProject,NotebookName):

    a = JupyterRoot
    b = " nbconvert --to script "
    c = Root
    d = FolderProject
    e = "/"
    f = NotebookName
    g = ".ipynb"
    line1 = a+b+c+d+e+f+g

    a = PythonRoot
    b = " "
    c = Root
    d = FolderProject
    e = "/"
    f = NotebookName
    g = ".py"
    line2 = a+b+c+d+e+f+g

    script = line1 + "\n" + line2 + "\n"

    return script


def WriteStringToFile(String,Path):    
    with open(Path, 'w') as out:
        out.write(String)
    return None


def WriteCleaningScript(RowsToRemove,Root,FolderProject,TweetsFilename):
    a = "tail -n +"
    b = str(RowsToRemove)
    c = " " + os.path.join(Root,FolderProject,TweetsFilename)
    d = " > "
    e = os.path.join(Root,FolderProject,TweetsFilename)
    commandbash = a+b+c+d+e
    return commandbash




def GetCurrentTime():
    now = datetime.now()
    current_time = now.strftime("%Y-%m-%d %H:%M:%S")
    return current_time


def SecondsToDelayFormat(SecondsElapsed):

    heure = np.floor(SecondsElapsed / 3600)
    heure = int(heure)
    reste = SecondsElapsed % 3600
    reste = np.floor(reste/60)
    reste  = int(reste)
    StringDuree = str(heure)+"h"+str(reste)+"min"    
    return StringDuree



def AppendStringToFile(path,toprint):
    with open(path,'a') as f:
        f.write(str(toprint)+"\n")
    return None


def RetrieveSize(path,formatted=True):
    
    Size = os.path.getsize(path)
    if formatted:
        Size = f'{Size:,}'
    
    return Size


def FormatNumber(Size):
    return f'{Size:,}'



def BuildStatsIrrelevant(tempdf,bassine_threshold,bassine_size):

    y = tempdf.groupby(["BassineID","AUTHORTWEETID"]).size().reset_index().rename(columns = {0:"f"})
    r = y.groupby('BassineID')['f'].rank(ascending=False).reset_index(drop=True)
    y["Rank"] = r
    y.sort_values(by=["BassineID","f"],ascending=False,inplace=True)
    y.reset_index(drop = True,inplace = True)
    y["cumsumf"] = y.groupby("BassineID")["f"].apply(lambda x: x.cumsum())
    sumdf = y.groupby("BassineID")["f"].sum().reset_index().rename(columns = {"f":"sum"})
    y = y.merge(sumdf,on="BassineID")
    y["p"] = (100*y.cumsumf) / y["sum"]
    y = y[y.p<bassine_threshold]
    y["BassineSize"]  = bassine_size

    return y


def ExtractQualifiedRT(RefRT,RefFam,bassine_size,bassine_recul,FirstBassineID):


    BassineID = RefFam.AUTHORTWEETUNIXEPOCH/bassine_size
    RefFam["BassineID"] = BassineID/bassine_size
    RefFam = RefFam[RefFam.BassineID>=FirstBassineID]

    df = RefRT.merge(RefFam[["AUTHORTWEETID","AUTHORTWEETUNIXEPOCH"]],on="AUTHORTWEETID")
    BassineID = df.AUTHORTWEETUNIXEPOCH/bassine_size
    df["BassineID"] = BassineID.astype(int)
    df["BassineSize"] = bassine_size
    df["age"] = df.TWEETUNIXEPOCH - df.AUTHORTWEETUNIXEPOCH

    BassineArray = df.BassineID.unique()
    BassineArray.sort()
    LastBassineID = BassineArray[-2]

    maxagedf = df.groupby("BassineID")["age"].max().reset_index()
    maxagedf = maxagedf[maxagedf.age>bassine_recul]
    maxagedf = maxagedf[maxagedf.BassineID>=FirstBassineID]
    maxagedf = maxagedf.rename(columns = {"age":"maxage"}).reset_index(drop = True)

    df = df.merge(maxagedf,on="BassineID")
    df = df[df.age<=bassine_recul]

    return df


def ExtractFirstBID(RefRT,OldISdf,bassine_size):
    
    if len(OldISdf)==0:
        datemin = RefRT.TWEETUNIXEPOCH.min()
        FirstBassineID = int(datemin/bassine_size)
    else:
        FirstBassineID = OldISdf.BassineID.max()+1
    
    return FirstBassineID







def DefineFirstLastDate(rtdf,StepSize,OldListOfTMResults):

    lastdate = rtdf.TWEETUNIXEPOCH.max()

    if len(OldListOfTMResults) == 0:
        firstdate = rtdf.TWEETUNIXEPOCH.min()
    else:
        LastTimeMark = max([v["TimeMark"] for k,v in OldListOfTMResults.items()])
        firstdate = LastTimeMark + StepSize

    return firstdate,lastdate




def BuildTweetsAuthorFromTimeMark(randomtm,WindowSize,StepSize,rtdf):
    
    periods = ExtractTheTwoPeriods(randomtm,WindowSize,StepSize)
    debut,fin = periods["TweetWindow"]
    myfilter = (rtdf.TWEETUNIXEPOCH>=debut) & (rtdf.TWEETUNIXEPOCH<=fin)
    temprtdf = rtdf.copy()[myfilter]
    temprtdf.reset_index(drop=True,inplace=True)
    
    # Modifier authoridname et tweetidname
    #TweetsAuthorsDF = temprtdf.groupby(authoridname)[tweetidname].apply(list).reset_index()
    TweetsAuthorsDF = temprtdf.groupby(["AUTHORID","AUTHORTWEETID"]).size().reset_index().rename(columns={0: "f"})
    
    
    return TweetsAuthorsDF



def BuildLinksDocFromTimeMark(randomtm,WindowSize,StepSize,rtdf):
    periods = ExtractTheTwoPeriods(randomtm,WindowSize,StepSize)
    debut,fin = periods["GraphWindow"]
    myfilter = (rtdf.TWEETUNIXEPOCH>=debut) & (rtdf.TWEETUNIXEPOCH<=fin)
    temprtdf = rtdf.copy()[myfilter]
    temprtdf.reset_index(drop=True,inplace=True)
    list_of_authors = temprtdf.groupby('USERID')['AUTHORID'].apply(list)
    LinksDic = GetLinksFromPeriod(list_of_authors)
    return LinksDic


def ListOfTMTreatment(ListOfTM,WindowSize,StepSize,rtdf,verbose=False):

    i = 1
    totali = len(ListOfTM)
    TimeMarkResultsDic = {}
    for randomtm in ListOfTM:
        if verbose:
            print("Period : ",str(i),"/",str(totali))
        i = i + 1
        
        LinksDic = BuildLinksDocFromTimeMark(randomtm,WindowSize,StepSize,rtdf)
        TweetsAuthorsDF = BuildTweetsAuthorFromTimeMark(randomtm,WindowSize,StepSize,rtdf)
        TimeMarkResultsDic[randomtm] = {"Links":LinksDic,
                                        "Tweets":TweetsAuthorsDF,
                                        "TimeMark":randomtm,
                                        "LinksWindow":(randomtm - WindowSize,randomtm),
                                        "TweetsWindow":(randomtm - StepSize,randomtm)}

    return TimeMarkResultsDic




def GetLinksFromPeriod(list_of_authors):
    
    L = []
    for SetOfIds in list_of_authors:
        if len(SetOfIds)>1:
            L = L + ExtractCleanLinksFromSets(SetOfIds)
    
    linkscompteur = {}
    for item in L:
        linkscompteur[item[0]] = linkscompteur.get(item[0],0) + item[1]
    
    return linkscompteur



def ExtractCleanLinksFromSets(SetOfIds):
    SetOfIds = [int(item) for item in SetOfIds]
    RawLinks = ExtractRawLinks(SetOfIds)
    CleanLinks = [ExplicitLink(item) for item in RawLinks]
    return CleanLinks

def ExtractRawLinks(SetOfIds):
    
    x = np.array(SetOfIds)
    comptage = np.unique(x,return_counts=True)
    list_of_tuples = np.vstack(comptage).T.tolist()
    combinations = list(itertools.combinations(list_of_tuples,2))
    
    return combinations

def ExplicitLink(x):
    weight = min([item[1] for item in x])
    link = tuple(item[0] for item in x)
    return link,weight


def DefineTimeMarks(lastdate,firstdate,step,window):

    mark0 = firstdate + window
    start = mark0
    L = [start]

    while(True):
        newmark = start+step
        if newmark>lastdate:
            break
        else:
            L.append(newmark)
            start=newmark

    return L


def ExtractTheTwoPeriods(timemark,WindowSize,StepSize):
    res = {"GraphWindow":(timemark-WindowSize,timemark),
           "TweetWindow":(timemark-StepSize,timemark)}
    return res

















def getbesttweets(rtdf_period,pbest):
    """
    input : un dataframe de retweets d'une période délimité
    output : un dataframe avec les meilleurs tweets retweetés sur la même période délimitée
    Pandas series des retweets provenant des tweets représenant p du volume émis
    """
    
    if rtdf_period is not None:  
        statsdf = rtdf_period.groupby("AUTHORTWEETID").size().reset_index().rename(columns = {0:"f"})
        statsdf = statsdf.sort_values(by = "f",ascending = False)
        statsdf["cumf"] = statsdf.f.cumsum()
        statsdf["proportion"] = statsdf.cumf/statsdf.f.sum()
        res = statsdf[statsdf.proportion<pbest]
        res = res.AUTHORTWEETID
    else:
        res = None
    
    return res



def getnexttm(rtdf,tmdic,stepsize,windowsize,verbose = False):
    """
    input : un dictionnaire de timemark
    input : le dataframe de retweets reference
    input : le stepsize
    output : soit un timemark s'il existe, soit un None Type
    """
    
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





def buildfilterdf(rtdf,nexttm,windowsize):
    """
    input : (rtdf) : le dataframe de retweets
    input : (tmdic) : le dico de timemark
    input : (windowsize) : la taille de la fenetre glissante
    input : (stepsize) : le step de la fenetre glissante
    output : le dataframe filtré rt en fonction de nexttm, s'il existe
    """
    
    # Si nexttm existe alors on extrait le bash
    # Sinon on ressort un None Type
    if nexttm is None:
        return None
    else:
        borneinf = nexttm - windowsize
        bornesup = nexttm
        fil = (rtdf.TWEETUNIXEPOCH>borneinf) & (rtdf.TWEETUNIXEPOCH<bornesup)
        res = rtdf[fil]
        res.reset_index(drop=True,inplace=True)
        return res




def retrievertfromid(rtdf,tweetsid):
    """
    input rtdf : le dataframe de retweets
    input tweetsid : une pandas series de tweet id à récupérer
    output : un dataframe de retweets les meilleurs sur le batch d'après
    """
    if tweetsid is not None:
        tweetsdf = tweetsid.to_frame(name="AUTHORTWEETID")
        res = tweetsdf.merge(rtdf,on="AUTHORTWEETID")
        res.reset_index(drop=True,inplace=True)
    else:
        res = pd.DataFrame()
    return res





def getbestrtbashdic(rtdf,tmdic,stepsize,windowsize,pbest,RefFam,RefInf):
    """
    input : le dataframe de retweets
    input : le dictionnaire de timemark
    input : le step et la window de la fenetre glissante
    input : p, la proportion de tweets à garder
    output : le dataframe des rt sélectionnés, les plus représentatifs du next batch
    """
    
    nexttm = getnexttm(rtdf,tmdic,stepsize,windowsize)
    filterdf = buildfilterdf(rtdf,nexttm,windowsize)
    tweetsid = getbesttweets(filterdf,pbest)
    bestrtdf =  retrievertfromid(filterdf,tweetsid)
    

    if len(bestrtdf)>0:
        bestfamdf = pd.Series(bestrtdf.AUTHORTWEETID.unique()).to_frame(name="AUTHORTWEETID").merge(RefFam,on="AUTHORTWEETID")
        bestinfdf = pd.Series(bestrtdf.AUTHORID.unique()).to_frame(name="AUTHORID").merge(RefInf,on="AUTHORID")
        tmdic[nexttm] = {"tm" : nexttm,
                         "wholertsize" : len(filterdf),
                         "bestrtsize" : len(bestrtdf),
                         "bestfamsize" : len(bestfamdf),
                         "bestinfsize" : len(bestinfdf),
                         
                         "ntweets" : len(tweetsid),
                         "start" : str(pd.to_datetime(nexttm - windowsize, unit="s")),
                         "end" : str(pd.to_datetime(nexttm,unit="s")),
                         
                         "bestrt" : bestrtdf,
                         "bestfam": bestfamdf,
                         "bestinf":bestinfdf}
    
    
    return bestrtdf,tmdic






def getbestrtbashdic(rtdf,tmdic,stepsize,windowsize,pbest,RefFam,RefInf):
    """
    input : le dataframe de retweets
    input : le dictionnaire de timemark
    input : le step et la window de la fenetre glissante
    input : p, la proportion de tweets à garder
    output : le dataframe des rt sélectionnés, les plus représentatifs du next batch
    """
    
    nexttm = getnexttm(rtdf,tmdic,stepsize,windowsize)
    filterdf = buildfilterdf(rtdf,nexttm,windowsize)
    tweetsid = getbesttweets(filterdf,pbest)
    bestrtdf =  retrievertfromid(filterdf,tweetsid)
    

    if nexttm is not None:
        bestfamdf = pd.Series(bestrtdf.AUTHORTWEETID.unique()).to_frame(name="AUTHORTWEETID").merge(RefFam,on="AUTHORTWEETID")
        bestinfdf = pd.Series(bestrtdf.AUTHORID.unique()).to_frame(name="AUTHORID").merge(RefInf,on="AUTHORID")
        informations = {"tm" : nexttm,
                        "wholertsize" : len(filterdf),
                        "bestrtsize" : len(bestrtdf),
                        "bestfamsize" : len(bestfamdf),
                        "bestinfsize" : len(bestinfdf),
                        "ntweets" : len(tweetsid),
                        "start" : str(pd.to_datetime(nexttm - windowsize, unit="s")),
                        "end" : str(pd.to_datetime(nexttm,unit="s"))}
    
    else:
        bestfamdf = None
        bestinfdf = None
        informations = None
        bestrtdf = None
        
    
    return bestrtdf,bestfamdf,bestinfdf,informations,nexttm






def extractkeepremove(RefRT,threshold):
    
    if threshold is not None:
        RemoveRT = RefRT[RefRT.TWEETUNIXEPOCH<threshold]
        KeepRT = RefRT[RefRT.TWEETUNIXEPOCH>=threshold]
    else:
        RemoveRT = pd.DataFrame()
        KeepRT = RefRT
    
    return RemoveRT,KeepRT




def BuildNewsRefs(RemoveRT,FinalFam,FinalInf,RefInf,RefFam):
    
    if len(RemoveRT)>0:
        TWEETIDStoremove = extracttweetidstoremove(RemoveRT,RefFam,FinalFam)
        AUTHIDStoremove = extractinfidstoremove(RemoveRT,RefInf,FinalInf)
        NewRefFam = RefFam[~RefFam.AUTHORTWEETID.isin(TWEETIDStoremove)]
        NewRefInf = RefInf[~RefInf.AUTHORID.isin(AUTHIDStoremove)]
    else:
        NewRefFam = RefFam
        NewRefInf = RefInf
    
    return NewRefFam,NewRefInf










def retrievethreshold(tmdic,WindowSize,RefRT):

    tmlist = np.array(list(tmdic.keys()))

    if len(tmdic)>0:
        borneinf = max(tmlist) - WindowSize
        if borneinf>RefRT.TWEETUNIXEPOCH.min():
            positivearray = tmlist[borneinf - np.array(tmlist) >= 0]
            if len(positivearray)>0:
                return positivearray.max()

    return None


def extracttweetidstoremove(RemoveRT,RefFam,FinalFam):
    tweetsidsofremovert = pd.Series(RemoveRT.AUTHORTWEETID.unique()).to_frame(name="AUTHORTWEETID").merge(RefFam,on="AUTHORTWEETID").AUTHORTWEETID.unique()
    tweetsidsoffinal = FinalFam.AUTHORTWEETID.unique()
    removethoseids = np.setdiff1d(tweetsidsofremovert,tweetsidsoffinal)
    return removethoseids 



def extractinfidstoremove(RemoveRT,RefInf,FinalInf):
    authidsofremovert = pd.Series(RemoveRT.AUTHORID.unique()).to_frame(name="AUTHORID").merge(RefInf,on="AUTHORID").AUTHORID.unique()
    authoffinal = FinalInf.AUTHORID.unique()
    removethoseids = np.setdiff1d(authidsofremovert,authoffinal)
    return removethoseids