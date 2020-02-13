
import json
import os

Root = '/home/osboxes/proj/streamer/'
FolderProject ='trackVirus'

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
