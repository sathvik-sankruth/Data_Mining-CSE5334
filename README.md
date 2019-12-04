# Term Project Data Mining
# Phase1: Search

A search feature which ranks the result based on the tf-idf scores.

Website Link:http://satsankruth.pythonanywhere.com/ <br>
Dataset Link: https://www.kaggle.com/datasnaek/youtube-new

**How to Deploy code:**
Create an account in pythonanywhere <br>
Create a virtual environment and install python3, flask, nltk, numpy, pandas <br>
Add the files <br>
Go to the http link <br>

**How code works:**
Read data which is a csv file from pandas.  <br>
Preprocess data to lower case. <br>
Remove punctuations and apostrophe. <br>
Remove stop words using nltk. <br>
Stem the data using Porter Stemming. <br>
Calculate term frequency. <br>
Calculate Inverse Document Frequency. <br>
Store tf-idf in a new pickle file which saves lot of time. <br>

Query Words: <br>
The query input is taken from the html page. <br>
Process the query convert it to lower case. <br>
Remove stop words from query. <br>
Rank the documents based on the similarity. <br>
The top 20 results are displayed. <br>

**Calculating TF-IDF:**

Term Frequency(TF)=Number of times the word occures in document / Total Number of words in document

Inverse Document Frequency(idf)=log( Total Number of documents / Number of documents with the keyword )

TF_IDF= TF * IDF


**Challenges faced:**
The data set is huge which has around 40K rows. Takes a long time for more than 40 seconds to read the data set and perform calculations for each query. Hence Im reading the dataset at the beginning of the search function and precalculating the tf-idf scores, so that it's easy to get the search result once the page is loaded.

**Contributions:**
Implemented stop words reduction and stemming while calculating tf-idf. <br>
Added pickle to save the initial loading time. <br>



**References:**
* https://github.com/williamscott701/Information-Retrieval/blob/master/2.%20TF-IDF%20Ranking%20-%20Cosine%20Similarity%2C%20Matching%20Score/TF-IDF.ipynb

* https://www.geeksforgeeks.org/removing-stop-words-nltk-python/

* https://help.pythonanywhere.com/pages/Flask/

* https://www.youtube.com/watch?v=M-QRwEEZ9-8




# Phase 2
Classifier: Multinomial Naïve Bayes Classifier<br>
Website Link : http://satsankruth.pythonanywhere.com/ <br>

A classifier is an algorithm that separates similar objects based on some features. For example, A video can be classified into music, autos & vehicles, comedy, gaming based on the category. However, what if we want to classify a new video? This can be done using very simple concept of probability, which is Naïve Bayes. Naïve Bayes is one of the popular classifiers when it comes to text classification. We can calculate the conditional probability of each term in the dataset’s vocabulary appear in each category.

**1. Preprocessing:** <br>
Because the data set has many repeated items with 40K rows, data set was modified so that it has only unique items which was reduced to 6K rows. 

![img](https://sathvik-sankruth.netlify.com/img/class1.PNG)

After saving the new data set, we convert to lowercase, perform stemming and eliminate stop words to construct a full vocabulary list.
 
**2. Processing:** <br>
Naïve Bayes classification is implemented using the formula.

![img](https://sathvik-sankruth.netlify.com/img/Eqn.png)

Therefore, the mission is to construct a table to store the conditional probability for each term in the vocabulary with respect to each category.

![img](https://sathvik-sankruth.netlify.com/img/class2.PNG)

However, first, for each class c in the categories, the prior probability would be the number of documents in class c divided by the total number of documents.

![img](https://sathvik-sankruth.netlify.com/img/Eqn2.png)

Then the conditional probability can be calculated by the frequency of term t in document belong to class c.

To avoid sparseness that would lead conditional probabilities to be zero, we should apply smoothing into the calculation. To do this, we add 1 to all the term to avoid 0 on the numerator and add the number of unique terms to the denominator for normalization.
 
**3. Classification:** <br>
1)query is given for classification <br>
2) preprocess the query  Stemming, and eliminate Stop Words. <br>
3) A score will be given for each class using Naïve Bayes. <br>
4)The final step is to sort the score. <br>

![img](https://sathvik-sankruth.netlify.com/img/class3.PNG)

**4. Contribution:** <br>
To avoid  sparseness which leads probabilities to be zero, Applied Smoothing hyperparameter to 1. <br>
![img](https://sathvik-sankruth.netlify.com/img/class6.PNG)
Selected the hyperparameter to 0.1 because it has the highest accuracy rate. As we can see in the graph the accuracy rate drops down when the hyperparameter value is more <br>

The Data set was split into Test data and Train data from sklearn.model_selection which ensures even distribution of the data. <br>

![img](https://sathvik-sankruth.netlify.com/img/class4.PNG)

Compared the Naïve Bayes classifier and SVM and found that Naive Bayes classifier works better for my data set. <br>

![img](https://sathvik-sankruth.netlify.com/img/class5.PNG)
 
Shows the percentages of classes. <br>
 
**5. Challenges faced:** 
Optimizing the algorithm was a bit challenging. <br>
The data set has many repeated items with 40K rows. Modified data set so that it has only unique items, Hence dataset was reduced to 6K rows.  <br>
 


**Reference:** <br>

* https://medium.com/@bedigunjit/simple-guide-to-text-classification-nlp-using-svm-and-naive-bayes-with-python-421db3a72d34

* https://towardsdatascience.com/naive-bayes-document-classification-in-python-e33ff50f937e

* https://nlp.stanford.edu/IR-book/html/htmledition/naive-bayes-text-classification-1.html

* https://nlp.stanford.edu/IR-book/pdf/13bayes.pdf



# Phase 3 Image Caption

Dataset Link: https://www.kaggle.com/hsankesara/flickr-image-dataset <br>
Website Link: http://satsankruth.pythonanywhere.com/ <br>
Demo Video Link: https://www.youtube.com/watch?v=LwQrrgDDlmE <br>

Image captioning is the task of generating a caption for an image.
Uses TensorFlow and Neural Network to generate captions on the google collab repository.
This model architecture is similar to Neural Image Caption Generation with Visual Attention.
The code uses tf.keras and eager execution
The notebook will download the MS-COCO dataset, preprocess and cache a subset of the images using Inception V3, train an encoder-decoder model, and use it to generate captions on new images.
 <br>
 
**Contribution:**<br>
Modified the code to test my dataset and stored it in csv file with image url and captions.<br>
![img](https://sathvik-sankruth.netlify.com/img/imgcap1.PNG)
![img](https://sathvik-sankruth.netlify.com/img/imgcap2.PNG)
 <br>
**Challenges Faced:**<br>
Training the model takes around 3hrs.<br>
My dataset was huge 30K images which is around 4Gb hence modified my dataset to 2K images which is around 600Mb.<br>
Testing the model with the reduced data took additional 1hr.<br>
Understanding how Jupyter Notebook, Tensor flow, Neural Network, Google Collab Feature works was a bit challenging.<br>

**Limitations:**<br>
There are a few images for which the generated captions failed.<br>
![img](https://sathvik-sankruth.netlify.com/img/imgcap3.PNG)
![img](https://sathvik-sankruth.netlify.com/img/imgcap4.PNG)
**Reference:**<br>

* https://github.com/tensorflow/tensorflow/blob/r1.13/tensorflow/contrib/eager/python/examples/generative_examples/image_captioning_with_attention.ipynb

* https://arxiv.org/abs/1502.03044

* https://www.tensorflow.org/guide/keras

* https://www.tensorflow.org/guide/eager

* https://hackernoon.com/begin-your-deep-learning-project-for-free-free-gpu-processing-free-storage-free-easy-upload-b4dba18abebc 


