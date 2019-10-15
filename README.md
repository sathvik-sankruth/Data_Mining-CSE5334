# phase1-search

A search feature which ranks the result based on the tf-idf scores.

Website Link:http://sankruth06.pythonanywhere.com/
Github Link: 

# How to Deploy code :
Create an account in pythonanywhere
Create a virtual environment and install python3, flask, nltk, numpy, pandas
Add the files
Go to the http link

# How code works:
Read data which is a csv file from pandas. 
Tokenizer uses regular expressions to remove unwanted data.
Remove stop words using nltk.
Calculate term frequency
Calculate Inverse Document Frequency
Store tf-idf

Query Words:
The query input is taken from the html page
Process the query convert it to lower case.
Remove stop words from query.
Rank the documents based on the similarity
The top 20 results are displayed.

# Calculating TF-IDF:

Term Frequency(TF)=Number of times the word occures in documentTotal Number of words in document

Inverse Document Frequency(idf)=Total Number of documentsNumber of documents with the keyword

TF_IDF=TF*IDF


# Challenges faced:
The data set is huge which has around 40K rows. Takes a long time for more than 40 seconds to read the data set and perform calculations for each query. Hence Im reading the dataset at the beginning of the search function and precalculating the tf-idf scores, so that it's easy to get the search result once the page is loaded.

# Contributions:
Implemented stop words reduction while calculating tf-idf.



# References:
https://github.com/williamscott701/Information-Retrieval/blob/master/2.%20TF-IDF%20Ranking%20-%20Cosine%20Similarity%2C%20Matching%20Score/TF-IDF.ipynb

https://www.geeksforgeeks.org/removing-stop-words-nltk-python/

https://help.pythonanywhere.com/pages/Flask/


