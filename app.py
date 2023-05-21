from flask import Flask,render_template,request,jsonify


import pickle
import pandas as pd


#load  model
model=pickle.load(open('model.pkl','rb'))
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/desc')
def description():
    return render_template('description.html')


@app.route('/form_data', methods=['POST', 'GET'])
def form():
    if request.method == 'POST':
        data = str(request.form.get('textarea', ''))
        data="\""+data+"\""
        a=model.predict([data])
        if a[0]==0:
            a="Not spam"
        else:
            a="Spam"    
        print(a)
        
    return render_template('response.html', data=a)                                               


if __name__ == '__main__':
    app.run(debug=True)