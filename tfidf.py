from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from collections import Counter
#from num2words import num2words

import nltk
import os
import string
import numpy as np
import copy
import pandas as pd
import pickle
import re
import math

def convert_lower_case(data):
    return np.char.lower(data)

def remove_stop_words(data):
    stop_words = stopwords.words('english')
    words = word_tokenize(str(data))
    new_text = ""
    for w in words:
        if w not in stop_words and len(w) > 1:
            new_text = new_text + " " + w
    return new_text

def remove_punctuation(data):
    symbols = "!\"#$%&()*+-./:;<=>?@[\]^_`{|}~\n"
    for i in range(len(symbols)):
        data = np.char.replace(data, symbols[i], ' ')
        data = np.char.replace(data, "  ", " ")
    data = np.char.replace(data, ',', '')
    return data

def remove_apostrophe(data):
    return np.char.replace(data, "'", "")


def stemming(data):
    stemmer = PorterStemmer()

    tokens = word_tokenize(str(data))
    new_text = ""
    for w in tokens:
        new_text = new_text + " " + stemmer.stem(w)
    return new_text

def preprocess(data):
    data = convert_lower_case(data)
    data = remove_punctuation(data) #remove comma seperately
    data = remove_apostrophe(data)
    data = remove_stop_words(data)
    #data = convert_numbers(data)
    data = stemming(data)
    data = remove_punctuation(data)
    return data

def invertedindex():
    video_Data = pd.read_csv("youtube-new/USvideos.csv")
    processed_text=[]
    processed_title=[]
    for i in range(len(video_Data)):
        processed_text.append(word_tokenize(str(preprocess(video_Data['title'][i]))))
        processed_title.append(word_tokenize(str(preprocess(video_Data['tags'][i]))))


    print(processed_text[0])
    DF = {}
    N=len(video_Data)
    for i in range(N):
        tokens = processed_text[i]
        for w in tokens:
            try:
                DF[w].add(i)
            except:
                DF[w] = {i}
        tokens = processed_title[i]
        for w in tokens:
            try:
                DF[w].add(i)
            except:
                DF[w] = {i}
    for i in DF:
        DF[i] = len(DF[i])

    print(len(DF))

    def doc_freq(word):
        c = 0
        try:
            c = DF[word]
        except:
            pass
        return c


    doc = 0

    tf_idf = {}

    for i in range(N):

        tokens = processed_text[i]

        counter = Counter(tokens + processed_title[i])
        words_count = len(tokens + processed_title[i])

        for token in np.unique(tokens):
            tf = counter[token] / words_count
            df = doc_freq(token)
            idf = np.log((N + 1) / (df + 1))

            tf_idf[doc, token] = tf * idf

        doc += 1

    doc = 0

    tf_idf_title = {}

    for i in range(N):

        tokens = processed_title[i]
        counter = Counter(tokens + processed_text[i])
        words_count = len(tokens + processed_text[i])

        for token in np.unique(tokens):
            tf = counter[token] / words_count
            df = doc_freq(token)
            idf = np.log((N + 1) / (df + 1))  # numerator is added 1 to avoid negative values

            tf_idf_title[doc, token] = tf * idf

        doc += 1


    alpha=0.3
    for i in tf_idf:
        tf_idf[i] *= alpha

    for i in tf_idf_title:
        tf_idf[i] = tf_idf_title[i]

    pickle.dump(tf_idf, open("save.p", "wb"))


    #print(len(tf_idf))


def matching_score(k, query):
    preprocessed_query = preprocess(query)
    tokens = word_tokenize(str(preprocessed_query))
    video_Data = pd.read_csv("youtube-new/USvideos.csv")
    tf_idf = pickle.load(open("save.p", "rb"))

    print("Matching Score")
    print("\nQuery:", query)
    print("")
    print(tokens)

    query_weights = {}

    for key in tf_idf:

        if key[1] in tokens:
            try:
                query_weights[key[0]] += tf_idf[key]
            except:
                query_weights[key[0]] = tf_idf[key]

    query_weights = sorted(query_weights.items(), key=lambda x: x[1], reverse=True)

    print("")

    l = []

    #for i in query_weights[:10]:
    #    l.append(i[0])

    #print(l)
    k = []

    for i in query_weights[:20]:
        l.append(i[0])
        k.append(i[1])

    out = []
    j = 0
    for i in l:
        out.append([video_Data['title'][i], video_Data['channel_title'][i],
                    video_Data['trending_date'][i], video_Data['thumbnail_link'][i], video_Data['tags'][i], k[j]])
        j += 1
    pd.set_option('display.max_columns', -1)
    pd.set_option('display.expand_frame_repr', False)
    pd.set_option('max_colwidth', -1)

    out = pd.DataFrame(out, columns=['title', 'channel title', 'Date', 'image', 'tags', 'tf*idf'])

    print(l)
    print("Done")
    return out


#matching_score(10, "trump")
