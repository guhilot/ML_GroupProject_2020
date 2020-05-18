import numpy as np
import pandas as pd
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import subjectivity
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import *
import matplotlib.pyplot as mlpt
import tweepy
import csv
import pandas as pd
import random
import numpy as np
import pandas as pd

##Read the data
data = pd.read_csv('Tesla_tweets_old.csv')

cdata=pd.DataFrame(columns=['Date','Tweets'])
index=0
for index,row in data.iterrows():
    stre=row["Tweets"]
    my_new_string = re.sub('[^ a-zA-Z0-9]', '', stre)
    cdata.sort_index()
    cdata.set_value(index,'Date',row["Date"])
    cdata.set_value(index,'Tweets',my_new_string)
    index=index+1
print(cdata.dtypes)

ccdata=pd.DataFrame(columns=['Date','Tweets'])

indx=0
get_tweet=""
for i in range(0,len(cdata)-1):
    get_date=cdata.Date.iloc[i]
    next_date=cdata.Date.iloc[i+1]
    if(str(get_date)==str(next_date)):
        get_tweet=get_tweet+cdata.Tweets.iloc[i]+" "
    if(str(get_date)!=str(next_date)):
        ccdata.set_value(indx,'Date',get_date)
        ccdata.set_value(indx,'Tweets',get_tweet)
        indx=indx+1
        get_tweet=" "

ccdata = pd.read_csv('Tesla_tweets_old.csv')
ccdata.head(200)

#Adding a "Price" column in our dataframe and fetching the stock price as per the date in our dataframe
ccdata['Prices']=""

read_stock_p=pd.read_csv('tesla_stock_price.csv')
read_stock_p

indx=0
for i in range (0,len(ccdata)):
    for j in range (0,len(read_stock_p)):
        get_tweet_date=ccdata.Date.iloc[i]
        get_stock_date=read_stock_p.Date.iloc[j]
        if(str(get_stock_date)==str(get_tweet_date)):
            #print(get_stock_date," ",get_tweet_date)
            ccdata.set_value(i,'Prices',int(read_stock_p.Close[j]))
            break


"""Filling empty price with previous day price"""
for i in range(1,len(ccdata)):
    if(ccdata.Prices.iloc[i]==""):
            ccdata.Prices.iloc[i]=int(ccdata.Prices.iloc[i-1])

ccdata['Prices'] = ccdata['Prices'].apply(np.int64)

ccdata["Comp"] = ''
ccdata["Negative"] = ''
ccdata["Neutral"] = ''
ccdata["Positive"] = ''
ccdata

import nltk
nltk.download('vader_lexicon')

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import unicodedata
sentiment_i_a = SentimentIntensityAnalyzer()
for indexx, row in ccdata.T.iteritems():
    try:
        sentence_i = unicodedata.normalize('NFKD', ccdata.loc[indexx, 'Tweets'])
        sentence_sentiment = sentiment_i_a.polarity_scores(sentence_i)
        ccdata.set_value(indexx, 'Comp', sentence_sentiment['compound'])
        ccdata.set_value(indexx, 'Negative', sentence_sentiment['neg'])
        ccdata.set_value(indexx, 'Neutral', sentence_sentiment['neu'])
        ccdata.set_value(indexx, 'Positive', sentence_sentiment['pos'])
    except TypeError:
        print (stocks_dataf.loc[indexx, 'Tweets'])
        print (indexx)

posi=0
nega=0
for i in range (0,len(ccdata)):
    get_val=ccdata.Comp[i]
    if(float(get_val)<(0)):
        nega=nega+1
    if(float(get_val>(0))):
        posi=posi+1
posper=(posi/(len(ccdata)))*100
negper=(nega/(len(ccdata)))*100
print("% of positive tweets= ",posper)
print("% of negative tweets= ",negper)
arr=np.asarray([posper,negper], dtype=int)
mlpt.pie(arr,labels=['positive','negative'])
mlpt.plot()

"""Adding Percent change column in the dataset"""
ccdata['percentchange'] = 0.0000
for i in range(0,len(cdata)-1):
    ccdata.at[i,'percentchange']=float((ccdata.Prices.iloc[i+1]-ccdata.Prices.iloc[i])/ccdata.Prices.iloc[i])

dataframe=ccdata[['Date','Prices','Comp','Negative','Neutral','Positive','percentchange']].copy()

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
plt.plot(date_test,y_test, label="actual")
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

print(mlpr.score(numpy_dataframe_train, train['Prices']))

date_test = np.array([x[0] for x in test1])
plt.plot(date_test,y_test, label="actual")
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
regr.score(numpy_dataframe_train,y_train)

date_test = np.array([x[0] for x in test1])
plt.plot(date_test,y_test, label="actual")
plt.plot(date_test,prediction_linear, label="Predicted")
plt.xlabel('Date') 
plt.ylabel('Price change')
plt.title('Linear Regression')
plt.legend()
plt.show()