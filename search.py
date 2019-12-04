from flask import Flask, render_template, url_for, request
import tfidf as ii
import os
app = Flask(__name__)


@app.before_first_request
def tdm_generator():
  if not os.path.isfile('imgtfidf.p'):
      ii.img()
  if not os.path.isfile('prior.p'):
      ii.something()
  if not os.path.isfile('save.p'):
    ii.invertedindex()


@app.route('/')
def home():
    return render_template('hello.html')


@app.route('/result', methods = ['POST','GET'])
def result():
  if request.method == 'POST':
    query = request.form['search']
    #simmat = be.similarity(str(query))
    #result = be.topk(simmat)
    result = ii.matching_score(20,str(query))
    #print(type(result))
    if type(result) != str:
        #print("in html")
        def path_to_image_html(path):
            return '<img src="' + path + '" width="120" >'
        return render_template("hello.html", query=query, result=result.to_html(escape=False,formatters=dict(image=path_to_image_html)))
    else:
      return render_template("hello.html", query=query, result=result)

@app.route('/classify',methods=['POST','GET'])
def classify():
    if request.method == 'POST':
        test=request.form['classify']
        categories = []
        percentage = []
        print("HELLO")
        if (test != None):
            classification,aprior,acond,tot = ii.classify(test)
            for c in classification:
                categories.append(c)
                percentage.append(round(classification[c] * 100, 2))
        #res=ii.classify(test)
        cat=len(categories)
        print("LEN OF CAT")
        print(cat)
        return render_template("hello.html",query2=test,categories = categories,percentage = percentage,aprior=aprior, acond=acond,tot=tot, classification = classification)


@app.route('/imgsearch', methods = ['POST','GET'])
def result1():
  if request.method == 'POST':
    query1 = request.form['imgsearch']
    #simmat = be.similarity(str(query))
    #result = be.topk(simmat)
    result1 = ii.imagesearch(10,str(query1))
    #print(type(result))
    if type(result1) != str:
        #print("in html")
        def path_to_image_html(path):
            return '<img src="' + path + '" width="120" >'
        return render_template("hello.html", query1=query1, result1=result1.to_html(escape=False,formatters=dict(image=path_to_image_html)))
    else:
        return render_template("hello.html", query1=query1, result1=result1)

if __name__=="__main__":
    app.run(debug=False)