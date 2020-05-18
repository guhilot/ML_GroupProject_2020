"""Initial Imports"""
import pandas as pd
import numpy as np
import re
import tweepy
import datetime
#from pandas_datareader import data as web
from textblob import TextBlob
from sklearn.svm import SVR
from treeinterpreter import treeinterpreter as ti
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

consumer_key = "W8olFrRLK7Lt7BUh6YAbdSG4h"
consumer_secret = "izRdiwC73rntSdMdAQm2gL8zFNAHVLa99A8dI99w0Lz16jvNrx"

access_token = "620364433-Kj5up6bB6KLfl5wfmxxcAnWthse5o38P2MKABM00"
access_token_secret = "Faxxv0hb54KF7MmakQ8WZmvaCC5YIprcWhiOuEyQY2iLt"

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth,wait_on_rate_limit=True)

fetch_tweets=tweepy.Cursor(api.search, q="#TSLA",count=100, lang ="en",since="2019-9-25", tweet_mode="extended").items()
data=pd.DataFrame(data=[[tweet_info.created_at.date(),tweet_info.full_text]for tweet_info in fetch_tweets],columns=['Date','Tweets'])

data.to_csv("TeslaTweets.csv")
cdata=pd.DataFrame(columns=['Date','Tweets'])
print("Created CSV")

##Read the data
data = pd.read_csv('tweets_data_old_elonmusk.csv')

cdata=pd.DataFrame(columns=['Date','Tweets'])
print(len(data))

index=0
for index,row in data.iterrows():
    stre=row["Tweets"]
    my_new_string = re.sub('[^ a-zA-Z0-9]', '', stre)
    cdata.sort_index()
    cdata.at[index,'Date'] =row["Date"]
    cdata.at[index,'Tweets']= my_new_string
    #index=index+1
ccdata=pd.DataFrame(columns=['Date','Tweets'])

#clubbed tweets date wise
indx=0
get_tweet=""
for i in range(0,len(cdata)-1):
    get_date=cdata.Date.iloc[i]
    next_date=cdata.Date.iloc[i+1]
    if(str(get_date)==str(next_date)):
        get_tweet=get_tweet+cdata.Tweets.iloc[i]+" "
    if(str(get_date)!=str(next_date)):
        ccdata.at[indx,'Date']=get_date
        ccdata.at[indx,'Tweets']=get_tweet
        indx+=1
        get_tweet=" "

ccdata = pd.read_csv('Tesla_tweets_old.csv')
ccdata.head()

"""Doing Sentiment Analysis of tweets"""

ccdata['polarity'] = 0.0000
ccdata['confidence'] =0.0000
for index,row in ccdata.iterrows():
    analysis = TextBlob(ccdata['Tweets'][index])
    sentiment, confidence  = analysis.sentiment
    ccdata.at[index,'polarity'] = sentiment
    ccdata.at[index,'confidence'] = confidence
    
posi=0
nega=0
for i in range (0,len(ccdata)):
    get_val=ccdata['polarity'][i]
    if(float(get_val)<(0)):
        nega=nega+1
    else:
        posi=posi+1
        
posper=(posi/(len(ccdata)))*100
negper=(nega/(len(ccdata)))*100
print("% of positive tweets= ",posper)
print("% of negative tweets= ",negper)
arr=np.asarray([posper,negper], dtype=int)
plt.pie(arr,labels=['positive','negative'])
plt.plot()

"""Get Stock Price From Yahoo using Pandas.dataReader"""

start = datetime.datetime(2012,10,26)
## Let's get Tesla stock data; Tesla's ticker symbol is TSLA
## First argument is the series we want, second is the source ("yahoo" for Yahoo! Finance), third is the start date, 
##fourth is the end date
tesla = web.DataReader('TSLA', 'yahoo', start)
#tesla.to_csv('TESLA_stock_price.csv')

read_stock_p=pd.read_csv('TESLA_stock_price.csv')
read_stock_p.head()

"""Add the stock price value matching tweets for the day"""

ccdata['Prices']=""
indx=0
for i in range (0,len(ccdata)):
    for j in range (0,len(read_stock_p)):
        get_tweet_date=ccdata.Date.iloc[i]
        get_stock_date=read_stock_p.Date.iloc[j]             
        if(str(get_stock_date)==str(get_tweet_date)):
            #print(get_stock_date," ",get_tweet_date)
            ccdata.at[i,'Prices']=int(read_stock_p.Close[j])
            break

#  """Filling empty price with previous day price"""
for i in range(1,len(ccdata)):
    if(ccdata.Prices.iloc[i]==""):
            ccdata.Prices.iloc[i]=int(ccdata.Prices.iloc[i-1])

#ccdata['Prices'] = ccdata['Prices'].apply(np.int64)


ccdata.head(6)
len(ccdata)

#"""Adding Percent change column in the dataset"""
ccdata['percentchange'] = 0.0000
print(len(ccdata))
for i in range(len(ccdata)-1,0,-1):
    ccdata.at[i-1,'percentchange']=float((ccdata.Prices.iloc[i-1]-ccdata.Prices.iloc[i])/ccdata.Prices.iloc[i-1])

#ccdata['percent change'] = ccdata['percent change'].apply(np.float)
ccdata.head()

ccdata.to_csv("Tesla_tweets_with_stocks_old.csv")

ccdata.head()

dataframe=ccdata[['Date','Prices','polarity','confidence','percentchange']].copy()

#Divide data into train and test 
train_data_start = 200
train_data_end = 549
test_data_start = 0
test_data_end = 199
train = dataframe.iloc[train_data_start: train_data_end]
test = dataframe.iloc[test_data_start:test_data_end]

train.head()

train1 = train.to_numpy()
test1 = test.to_numpy()
numpy_dataframe_train= np.array([x[2:6] for x in train1])

numpy_dataframe_test= np.array([x[2:6] for x in test1])

list_of_sentiments_score = []
for date, row in train.T.iteritems():
    sentiment_score = np.asarray([dataframe.loc[date, 'polarity']])
    list_of_sentiments_score.append(sentiment_score)
numpy_dataframe_train = np.asarray(list_of_sentiments_score)

list_of_sentiments_score = []
for date, row in test.T.iteritems():
    sentiment_score = np.asarray([dataframe.loc[date, 'polarity']])
    list_of_sentiments_score.append(sentiment_score)
numpy_dataframe_test = np.asarray(list_of_sentiments_score)

y_train = pd.DataFrame(train['percentchange'])
#y_train=[91,91,91,92,91,92,91]
y_test = pd.DataFrame(test['percentchange'])

from sklearn.metrics import precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import accuracy_score

from treeinterpreter import treeinterpreter as ti
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import classification_report,confusion_matrix

rf = RandomForestRegressor()
rf.fit(numpy_dataframe_train, y_train.values.ravel())
prediction_tree, bias, contributions = ti.predict(rf, numpy_dataframe_test)
prediction_rf = rf.predict(numpy_dataframe_test)
rf.score(numpy_dataframe_train,y_train.values.ravel())

rf.score(numpy_dataframe_test,y_test)

import matplotlib.pyplot as plt

date_test = np.array([x[0] for x in test1])
labels= ['sep17','aug17','jul17','jun17','may17','apr17','mar17','feb17',
         'jan17','dec16','nov16','oct16','sep16']
x_array = ['9/29/2017','8/31/2017','7/31/2017','6/30/2017','5/31/2017','4/29/2017','3/28/2017',
           '2/28/2017','1/29/2017','12/30/2016','11/24/2016','10/29/2016','9/13/2016']
plt.plot(date_test,y_test, label="actual")
plt.xticks(x_array,labels,rotation='vertical')
plt.plot(date_test,prediction_rf, label="Predicted")
plt.xlabel('Date') 
plt.ylabel('Price change')
plt.title('Random Forest')
plt.legend()
plt.show()

from sklearn.neural_network import MLPRegressor
mlpr = MLPRegressor(hidden_layer_sizes=(10,), activation='relu', 
                     solver='lbfgs', alpha=0.005, learning_rate_init = 0.001)
mlpr.fit(numpy_dataframe_train, train['percentchange'])   
prediction_mlp = mlpr.predict(numpy_dataframe_test)

mlpr.score(numpy_dataframe_train, train['Prices'])

mlpr.score(numpy_dataframe_test, test['Prices'])

date_test = np.array([x[0] for x in test1])
labels= ['sep17','aug17','jul17','jun17','may17','apr17','mar17','feb17',
         'jan17','dec16','nov16','oct16','sep16']
x_array = ['9/29/2017','8/31/2017','7/31/2017','6/30/2017','5/31/2017','4/29/2017','3/28/2017',
           '2/28/2017','1/29/2017','12/30/2016','11/24/2016','10/29/2016','9/13/2016']
plt.plot(date_test,y_test, label="actual")
plt.xticks(x_array,labels,rotation='vertical')
plt.plot(date_test,prediction_mlp, label="Predicted")
plt.xlabel('Date') 
plt.ylabel('Price change')
plt.title('MLP Regressor')
plt.legend()
plt.show()

from sklearn import datasets
from datetime import datetime, timedelta
from sklearn.naive_bayes import GaussianNB
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

regr = linear_model.LinearRegression()
regr.fit(numpy_dataframe_train, train['percentchange'])   
prediction_linear = regr.predict(numpy_dataframe_test)
regr.score(numpy_dataframe_test,y_test)

date_test = np.array([x[0] for x in test1])
labels= ['sep17','aug17','jul17','jun17','may17','apr17','mar17','feb17',
         'jan17','dec16','nov16','oct16','sep16']
x_array = ['9/29/2017','8/31/2017','7/31/2017','6/30/2017','5/31/2017','4/29/2017','3/28/2017',
           '2/28/2017','1/29/2017','12/30/2016','11/24/2016','10/29/2016','9/13/2016']
plt.plot(date_test,y_test, label="actual")
plt.xticks(x_array,labels,rotation='vertical')
plt.plot(date_test,prediction_linear, label="Predicted")
plt.xlabel('Date') 
plt.ylabel('Price change')
plt.title('Linear Regression')
plt.legend()
plt.show()

