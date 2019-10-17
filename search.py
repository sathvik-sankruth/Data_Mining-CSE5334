from flask import Flask, render_template, url_for, request
import tfidf as ii
import os
app = Flask(__name__)


@app.before_first_request
def tdm_generator():
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

if __name__=="__main__":
    app.run(debug=False)