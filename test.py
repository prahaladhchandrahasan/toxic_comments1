import numpy as np
from flask import Flask, request, jsonify, render_template
from keras.models import load_model
import re
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pandas as pd 
import nltk
import pickle
nltk.download('stopwords')
from nltk.corpus import stopwords


def compute(s):
    data = [[s,'','']]
    
    test = pd.DataFrame(data, columns = ['comment_text','preprocess','preprocess1'])
    print(test)

    stop = stopwords.words('english')

    #train['preprocess'] = train.apply(lambda row: row['comment_text'].replace("\n"," "), axis=1) #removes new line character
    test['preprocess'] = test.apply(lambda row: row['comment_text'].replace("\n"," "), axis=1)

    #removes urls
    #train['preprocess']=train.apply(lambda row: re.sub('http://\S+|https://\S+', 'urls',row['preprocess']).lower(), axis=1)
    test['preprocess']=test.apply(lambda row: re.sub('http://\S+|https://\S+', 'urls',row['preprocess']).lower(), axis=1)

    #remove all non-alphanumeric values(Except single quotes)
    #rain['preprocess']=train.apply(lambda row: re.sub('[^A-Za-z\' ]+', '',row['preprocess']).lower(), axis=1)
    test['preprocess']=test.apply(lambda row: re.sub('[^A-Za-z\' ]+', '',row['preprocess']).lower(), axis=1)

    #remove stopwords as they occupy major chunk of the vocabulary
    #train['preprocess'] = train['preprocess'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
    test['preprocess'] = test['preprocess'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

    #removes all additional spaces
    #train['preprocess']=train.apply(lambda row: re.sub('  +', ' ',row['preprocess']).strip(), axis=1)
    test['preprocess']=test.apply(lambda row: re.sub('  +', ' ',row['preprocess']).strip(), axis=1)


    test["preprocess1"] = test.apply(lambda x: x["comment_text"] if len(x["preprocess"])==0 else x['preprocess'], axis=1)


    testd=test["preprocess1"]

    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    test_final = tokenizer.texts_to_sequences(testd)
    testd=pad_sequences(test_final, maxlen=200)


    model = load_model("lstm.h5")

    preds = model.predict(testd)
    lis = []
    for i in preds:
        for j in i:
            lis.append(j)

    dict={
        "toxic":lis[0],
        "severe_toxic":lis[1],
        "obscene":lis[2],
        "threat":lis[3],
        "insult":lis[4],
        "identity_hate":lis[5]
    }
    
    return dict
    

