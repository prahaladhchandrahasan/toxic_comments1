import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from test import *

app = Flask(__name__)
#model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=(['POST']))
def predict():
    '''
    For rendering results on HTML GUI
    '''
    text =  [x for x in request.form.values()]
    prediction = compute(text[0])

    


    return render_template('index.html', prediction_text='The values are {}'.format(prediction))



if __name__ == "__main__":
    app.run(debug=True)