import tokenizer as tokenizer
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from collections import Counter
import math
import pandas as pd
import numpy as np
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
import nltk





#def invertedindex():
video_Data = pd.read_csv("youtube-new/USvideos.csv")
tokenizer = RegexpTokenizer('[a-zA-Z0-9]+')
token = []
final_document = []
stops = stopwords.words('english')
stemmer = PorterStemmer()
for i in range(len(video_Data)):
        tokens = tokenizer.tokenize(video_Data['title'][i])
        tokens += tokenizer.tokenize(video_Data['channel_title'][i])
        tokens += tokenizer.tokenize(video_Data['tags'][i])

        final_tokens = []
        for token in tokens:
            token = token.lower()
            if token not in stops:
                token = stemmer.stem(token)
                final_tokens.append(token)

        final_document.append(final_tokens)

DF = {}

def doc_freq(word):
        c = 0
        try:
            c = DF[word]
        except:
            pass
        return c

for i in range(len(final_document)):
        tokens = final_document[i]
        for w in tokens:
            try:
                DF[w].add(i)
            except:
                DF[w] = {i}

for i in DF:
        DF[i] = len(DF[i])



doc = 0
N = len(final_document)
tf_idf = {}

for i in range(N):

        tokens = final_document[i]

        counter = Counter(tokens + final_document[i])
        words_count = len(tokens + final_document[i])

        for token in np.unique(tokens):
            tf = counter[token] / words_count
            df = doc_freq(token)
            idf = np.log((N + 1) / (df + 1))

            tf_idf[doc, token] = tf * idf

        doc += 1

    #p = open('videodata.pickel',"wb")
    #pickle.dump(tf_idf,p)
    #global tf_idf1
    #tf_idf1 = tf_idf

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

def preprocess(data):
        data = convert_lower_case(data)
        # data = remove_punctuation(data) #remove comma seperately
        # data = remove_apostrophe(data)
        data = remove_stop_words(data)
        return data



def matching_score(k, query):
        query = preprocess(query)
        #video_Data = pd.read_csv("youtube-new/USvideos.csv")
        tokens = tokenizer.tokenize(str(query))
        #p = open('videodata.pickel', "rb")
        #tf_idf1 = pickle.load(p)
        print("TFIDF LENGTH= ",len(tf_idf))
        print("Matching Score")
        print("\nQuery:", query)
        print("")
        print(list(tokens))

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
        k = []

        for i in query_weights[:20]:
            l.append(i[0])
            k.append(i[1])

        out = []
        j=0
        for i in l:
            out.append([video_Data['title'][i], video_Data['channel_title'][i],
                        video_Data['trending_date'][i],video_Data['thumbnail_link'][i],k[j]])
            j+=1
        pd.set_option('display.max_columns', -1)
        pd.set_option('display.expand_frame_repr', False)
        pd.set_option('max_colwidth', -1)


        out = pd.DataFrame(out, columns=['title', 'channel title', 'Date','image','tf*idf'])


        print(l)
        print("Done")
        return out
