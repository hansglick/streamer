import json
import sys
import time
from tweepy import OAuthHandler, Stream
from tweepy.streaming import StreamListener
from tweepy import API

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
        return False



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