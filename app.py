from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application=Flask(__name__)

app=application

## Route for a home page

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
            Administrative = int(request.form.get('Administrative')),
            Administrative_Duration = float(request.form.get('Administrative_Duration')),
            Informational = int(request.form.get('Informational')),
            Informational_Duration = float(request.form.get('Informational_Duration')),
            ProductRelated = int(request.form.get('ProductRelated')),
            ProductRelated_Duration = float(request.form.get('ProductRelated_Duration')),
            BounceRates = float(request.form.get('BounceRates')),
            ExitRates = float(request.form.get('ExitRates')),
            PageValues = float(request.form.get('PageValues')),
            SpecialDay = float(request.form.get('SpecialDay')),
            Month = float(request.form.get('Month')),
            OperatingSystems = int(request.form.get('OperatingSystems')),
            Browser = int(request.form.get('Browser')),
            Region = int(request.form.get('Region')),
            TrafficType = int(request.form.get('TrafficType')),
            Weekend = int(request.form.get('Weekend')),
            Returning_Visitor = int(request.form.get('Returning_Visitor'))
        )
            
        
        
        pred_df=data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline=PredictPipeline()
        print("Mid Prediction")
        results=predict_pipeline.predict(pred_df)
        print("after Prediction")
        return render_template('home.html',results=results[0])
    

if __name__=="__main__":
    app.run(host="0.0.0.0")        