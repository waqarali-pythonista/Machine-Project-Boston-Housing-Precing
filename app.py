import pickle 
from flask import Flask , request , app, jsonify,url_for, render_template
import numpy as np 

import pandas as pd 

#this is starting point of my application 
app=Flask(__name__)
#load the regmodel 
regmodel=pickle.load(open('regmodel.pkl','rb'))
scaler=pickle.load(open('scaling.pkl','rb'))

@app.route('/')
def home ():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])

def predict_api():
    data=request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1, -1))
    
    new_Data=scalar.transform(np.array(list(data.values())).reshape(1, -1))
    output=regmodel.predict(new_Data)
    print(output[0])
    return jsonify (output[0])

if __name__=="__main__":
       app.run(debug=True) 
    