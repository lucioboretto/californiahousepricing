import pickle
from flask import Flask,request,app,josonify,url_for,render_template
import numpy as np
import pandas as pd

app=Flask(__name__)
## Load the model
regmodel=pickle.load(open('regmodel.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')
@app.route('/predict_api',methods=['POST'])

def predict_api():
    data=request.json['data']
    print(data)
    new_data=[list(data.values())]
    print(new_data)
    output=regmodel.predict(new_data)
    print(output[0])
    return jsonify(output[0])