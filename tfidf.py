from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from collections import Counter
#from num2words import num2words
import math
import pandas as pd
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from collections import defaultdict
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

def preprocess1(data):
    data = convert_lower_case(data)
    data = remove_punctuation(data) #remove comma seperately
    data = remove_apostrophe(data)
    #data = remove_stop_words(data)
    #data = convert_numbers(data)
    #data = stemming(data)
    #data = remove_punctuation(data)
    return data

def count_document1(document, c):
	document_in_c = 0
	for doc in document:
		if c == doc:
			document_in_c += 1
	return document_in_c


def concatenate_text1(categories, document, c):
    text_in_c = []
    for i in range(len(document)):
        if c == categories[i]:
            text_in_c.extend(document[i])

    return text_in_c


def something():
    #print("GOOGLE")
    vid = pd.read_csv('google.csv')
    tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
    stops = stopwords.words('english')
    stemmer = PorterStemmer()
    total_words = []
    final_document = []
    weight_vectors = []
    vocabulary = []
    categories = []
    prior = {}
    condprob = defaultdict(dict)
    for i in range(len(vid)):
        tokens = tokenizer.tokenize(vid['title'][i])
        tokens += tokenizer.tokenize(vid['tags'][i])
        # Remove stop words
        final_tokens = []

        for token in tokens:
            token = token.lower()
            if token not in stops:
                token = stemmer.stem(token)
                final_tokens.append(token)
                if token not in vocabulary:
                    vocabulary.append(token)
        final_document.append(final_tokens)
    total_document = len(final_document)

    total_term = len(vocabulary)
    print("C_ID")
    ratings = vid['category_id']
    C_ID=ratings
    # categories = []
    for rating in ratings:
        if rating not in categories:
            categories.append(rating)

    for c in categories:
        # Count how many documents are in class c
        document_in_c = count_document1(C_ID, c)
        # print(document_in_c)
        prior[c] = document_in_c / float(total_document)
        # Concatenate all the text of class c in one list
        text_in_c = concatenate_text1(C_ID, final_document, c)

        for term in vocabulary:
            # Count how many term t are in class c
            Tct = text_in_c.count(term)
            condprob[term][c] = (Tct + 1) / (len(text_in_c) + total_term)

    pickle.dump(prior,open("prior.p", "wb"))
    pickle.dump(condprob, open("condprob.p", "wb"))
    pickle.dump(categories,open("categories.p", "wb"))

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

def img():
    img_Data = pd.read_csv("images.csv")
    processed_text1 = []
    # processed_title=[]
    for i in range(len(img_Data)):
        processed_text1.append(word_tokenize(str(preprocess1(img_Data['caption'][i]))))
    print(processed_text1[0])
    DF1 = {}
    N1 = len(img_Data)
    for i in range(N1):
        tokens = processed_text1[i]
        for w in tokens:
            try:
                DF1[w].add(i)
            except:
                DF1[w] = {i}
    for i in DF1:
        DF1[i] = len(DF1[i])

    total_vocab_size1 = len(DF1)
    print(total_vocab_size1)
    total_vocab1 = [x for x in DF1]
    print(total_vocab1[:20])

    def doc_freq1(word):
        c = 0
        try:
            c = DF1[word]
        except:
            pass
        return c

    doc1 = 0

    tf_idf1 = {}

    for i in range(N1):

        tokens = processed_text1[i]

        counter = Counter(tokens + processed_text1[i])
        words_count = len(tokens + processed_text1[i])

        for token in np.unique(tokens):
            tf = counter[token] / words_count
            df = doc_freq1(token)
            idf = np.log((N1 + 1) / (df + 1))

            tf_idf1[doc1, token] = tf * idf

        doc1 += 1

    print(len(tf_idf1))
    alpha = 0.3
    for i in tf_idf1:
        tf_idf1[i] *= alpha

    pickle.dump(tf_idf1, open("imgtfidf.p", "wb"))



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

def classify(test):
    #test = "trump india"
    catdict = {1: 'Film & Animation', 2: 'Autos & Vechicles', 10: 'Music',
               15: 'Pets & Animals', 17: 'Sports', 18: 'Short Movies', 19: 'Travel & Events',
               20: 'Gaming', 21: 'Videoblogging', 22: 'People & Blogs', 23: 'Comedy',
               24: 'Entertainment', 25: 'News & Politics', 26: 'Howto & Style',
               27: 'Education', 28: 'Science & Technology', 29: 'Nonprofits & Activism',
               30: 'Movies', 31: 'Anime/Animation', 32: 'Action/Adventure',
               33: 'Classics', 34: 'Comedy', 35: 'Documentary', 36: 'Drama',
               37: 'Family', 38: 'Foreign', 39: 'Horror', 40: 'Sci-Fi/Fantasy',
               41: 'Thriller', 42: 'Shorts', 43: 'Shows', 44: 'Trailers'}
    test_vocab = []
    print("IN CLASSIFY")
    categories = pickle.load(open("categories.p", "rb"))
    prior = pickle.load(open("prior.p", "rb"))
    condprob = pickle.load(open("condprob.p", "rb"))
    print("After load IN CLASSIFY")
    tokenizer = RegexpTokenizer(r'[a-zA-Z]+')

    stops = stopwords.words('english')

    stemmer = PorterStemmer()

    terms = tokenizer.tokenize(test)
    for term in terms:
        term = term.lower()
        if term not in stops:
            term = stemmer.stem(term)
            test_vocab.append(term)

    score = {}
    for c in categories:
        score[c] = prior[c]
        for term in test_vocab:
            if term in condprob:
                score[c] *= condprob[term][c]

    total_score = sum(score.values())
    # print(total_score)
    classification={}
    aprior=[]
    acond=[]
    #ascore=[]
    for c in sorted(score, key=score.get, reverse=True):
        #print(catdict[c])
        print("")
        #print("prior", prior[c])
        aprior.append(prior[c])
        #for t in test_vocab:
            #print("cond", condprob[t][c])
            #acond.append(condprob[t][c])
        #print("score", score[c])
        acond.append(score[c])
        #print("category", catdict[c])
        #print("percent", score[c] / float(total_score))
        #ascore.append(score[c] / float(total_score))
        classification[catdict[c]]=score[c] / float(total_score)
    # print(s.values()/sum(score.values()))
    return classification,aprior,acond,total_score

def imagesearch(k1,query):
    preprocessed_query = preprocess1(query)
    tokens = word_tokenize(str(preprocessed_query))
    img_Data = pd.read_csv("images.csv")
    tf_idf1 = pickle.load(open("imgtfidf.p", "rb"))

    query_weights = {}

    for key in tf_idf1:

        if key[1] in tokens:
            try:
                query_weights[key[0]] += tf_idf1[key]
            except:
                query_weights[key[0]] = tf_idf1[key]

    query_weights = sorted(query_weights.items(), key=lambda x: x[1], reverse=True)

    l = []

    k = []

    for i in query_weights[:10]:
        l.append(i[0])
        k.append(i[1])

    out1 = []
    j = 0
    for i in l:
        out1.append([img_Data['caption'][i], img_Data['url'][i],k[j]])
        j += 1
    pd.set_option('display.max_columns', -1)
    pd.set_option('display.expand_frame_repr', False)
    pd.set_option('max_colwidth', -1)

    out1 = pd.DataFrame(out1, columns=['caption', 'image','tf*idf'])

    print(l)
    print("Done")
    return out1
