#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer
import scipy

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense    # for hidden layers of tensorflow.NOT dense matrix

# Data cleaning libraries

import re            #regex . (helps in data cleaning)
import nltk          #stop words are junk in data. nltk has these stop words,punctuations, strings.
nltk.download('stopwords')
from nltk.corpus import stopwords

df = pd.read_csv("/content/drive/MyDrive/Classroom/Tweets.csv")
df.head()

df.drop("tweet_id",axis=1,inplace = True)
df.head()
df.isnull().sum()
df.shape

df.drop(['airline_sentiment_confidence','airline_sentiment_gold','negativereason_gold','tweet_coord','tweet_location','user_timezone'],inplace = True,axis=1)
df.head()

# Data analysis
df['airline_sentiment'].value_counts().plot(kind='bar')
df['airline'].value_counts().plot(kind='bar')
df.negativereason.value_counts().plot(kind='bar')
df_each_airline_sentiment = pd.crosstab(df.airline, df.airline_sentiment)
df_each_airline_sentiment
df_each_airline_sentiment.plot(kind='bar')
df_each_airline_neg_sentiment = pd.crosstab(df.airline, df.negativereason)
df_each_airline_neg_sentiment.style.highlight_max(color='red',axis=1)
df_each_airline_neg_sentiment.plot(kind='bar',figsize=(30,10))
df['tweet_created']=pd.to_datetime(df.tweet_created).dt.strftime('%d-%m-%y')
df.head()
df.tweet_created.max()
df.tweet_created.min()
df_negReason_date = pd.crosstab(df.tweet_created,df.airline_sentiment)
df_negReason_date

df_negReason_date.plot(kind='bar',figsize=(25,10))

#Data Cleaning

stops=set(stopwords.words('english'))
stops
len(stops)

def data_clean(sentence):
  words=sentence.lower()  #Converts all the capital letters to small letters
  words=re.sub('[^a-z]',' ',words)   # ^ means allowed
  words=words.split()   # it will create a list
  important_words=[w for w in words if w not in stops] # to filter imp word 
  return(' '.join(important_words))   #  to join into a sentence using space between words

ex='it is 9:40 but we are not tired we still want to learn deep learning we stay up till 11 up today'

data_clean(ex)

df['clean_text']=df.text.apply(lambda x : data_clean(x)) # lambda is applying cleaning function to each line
result=df[['text','clean_text']]
result

result['count_text'] = result['clean_text'].apply(lambda x : len(x.split()))    # count the no. of words in each text
result['count_text']

result.join(df['airline_sentiment'])  # joining a column from other dataframe

result['count_text'].min()
result['count_text'].max()
result['count_text'].plot(kind='hist')
result.drop(['text','count_text'],axis = 1, inplace=True)
result

new_df = result.join(df['airline_sentiment'])
ew_df
new_df = pd.get_dummies(new_df['airline_sentiment'],columns=['neutral','positive','negative'])   # one hot encoding required since we are looking at categorical cross entropy
new_df

encoded_df = result.join(new_df)

x = encoded_df['clean_text']
y = encoded_df.iloc[:,1:]
y.head()

# Train test split
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 100)

X_train.shape

X_test.shape

y_train.shape

y_test.shape
# using TF_IDF vectorizer

tfidf = TfidfVectorizer()
x_train_idf = tfidf.fit_transform(X_train)    # to get the dict(vocab) of words and values in training data
x_train_idf
tfidf.get_feature_names()
len(tfidf.get_feature_names())
x_test_idf = tfidf.transform(X_test)
x_test_idf

# Converting train data into dense matrix

X_train = scipy.sparse.csr_matrix.todense(x_train_idf)  # use scipy to have lesser run time
X_train
X_train.shape   # 11712 (rows) => no of neurons at input layer, 11888 (columns) vocabulary => input_shape

X_test = scipy.sparse.csr_matrix.todense(x_test_idf)   # tfidf accepts matrix only in dense form;else it gives a shape error
X_test.shape

# Model Building

neurons_first_layer = X_train.shape[0]

neurons_first_layer

input = X_train.shape[1]

input

output = y_train.shape[1]

output


model = Sequential([
                    Dense(neurons_first_layer,activation='relu',input_shape=(input,)),   # 'input' no.of columns and all the rows
                    #Dense(128,activation='relu'),
                    Dense(64,activation='relu'),
                    Dense(output,activation='softmax')
                   ])


model.summary()

model.compile(optimizer='adam',loss = 'categorical_crossentropy',metrics=['accuracy'])
history = model.fit(X_train,y_train,epochs = 5,validation_data = (X_test,y_test))

plt.plot(history.history['accuracy'])                #training accuracy
plt.plot(history.history['val_accuracy'])
plt.xlabel('epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy Curve Analysis')
plt.legend(['Training','Testing'],loc = 'best')

plt.plot(history.history['loss'])                #training accuracy
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('Loss Curve Analysis')
plt.legend(['Training','Testing'],loc = 'best')

model.save('Deep_Learning.h5')  #model.load to load the model later

import pickle

with open('tfidfVect.pickle','wb') as handle:   #save it in a notepad
  pickle.dump(tfidf,handle,protocol = pickle.HIGHEST_PROTOCOL)

test = ['the flight was safe']

x_test_idf = tfidf.transform(test)
X_test = scipy.sparse.csr_matrix.todense(x_test_idf)
new_model = tf.keras.models.load_model('./Deep_Learning.h5')

pred = new_model.predict(X_test)

pred

category = {'Negative':0,'Neutral':1,'Positive':2}

labels = list(category.keys())
labels

labels[np.argmax(pred)]
