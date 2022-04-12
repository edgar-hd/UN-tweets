import os
import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
import snscrape.modules.twitter as sntwitter
from tqdm import tqdm

# Use snscrape to collect tweets, more useful for this than the official twitter API as one isn't restricted in
# how many tweets to download, etc.
def get_tweets(init_date, end_date, query):
    tweets_list = []
    # Scrape data and append tweets to list
    for i,tweet in enumerate(sntwitter.TwitterSearchScraper(
        query + ' since:'+init_date+' until:'+end_date+'').get_items()):
        if i>1000000:
            print("Error: too many tweets")
            break
        tweets_list.append([tweet.date, tweet.id, tweet.content, tweet.user.username, tweet.user.id, tweet.lang])
    # Labelling
    tweets_df = pd.DataFrame(tweets_list, columns=['Datetime', 'Tweet Id', 'Text', 'Username', 'User Id', 'Language'])
    
    return(tweets_df)

    # Set up regular time intervals to download tweets, just useful to cut up the data as one can then collect more in
# the future or past, also in case there's an error it would only affect one file out of many
def get_date_range(start_date,sectSize):
    # start_date = '20210101'
    date_format = '%Y%m%d'
    dtObj = datetime.strptime(start_date, date_format)
    numDays = datetime.today().date()-dtObj.date()
#     sectSize = 10
    numSections = np.floor(numDays.days/sectSize).astype(np.int16)
    
    # Make sure data is properly labelled by time for future reference and to incorporate with more data
    dateList = [(dtObj + relativedelta(days=i*sectSize)).date() for i in range(numSections)]
    dateList_string = [dateList[i].strftime(date_format)+'-'
                       +(dateList[i]+relativedelta(days=sectSize-1)).strftime(date_format) for i in range(len(dateList))]
    return(dateList,dateList_string)

# Collect and save the tweets into CSV
def collect_save_tweets(dateList, dateList_string,query,sectSize):
    for i in tqdm(range(len(dateList))):
        sect_tweets = get_tweets(str(dateList[i]),str(dateList[i]+relativedelta(days=sectSize)),query)
        sect_tweets.to_csv('data_tweets/un_mentions_'+dateList_string[i]+".csv",
                            index=False)

if not os.path.exists('data_tweets'): os.makedirs('data_tweets')
# This takes about 2 hours for approx 400 days, roughly 200 days/hour
dateList, dateList_string = get_date_range('20220317', 10)
collect_save_tweets(dateList, dateList_string, "#UN", 10)