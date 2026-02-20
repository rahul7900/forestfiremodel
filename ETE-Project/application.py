import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from flask import Flask, request, jsonify, render_template

application = Flask(__name__)
app =  application

## import piclked model
ridge_model = pickle.load(open(r'E:\Data Science_Study\Python\Python\Complete-Python-Bootcamp-main\29-End_to_End_LR_Project\ETE-Project\models\Ridge.pkl','rb'))
standard_scalar = pickle.load(open(r'E:\Data Science_Study\Python\Python\Complete-Python-Bootcamp-main\29-End_to_End_LR_Project\ETE-Project\models\Scalar.pkl','rb'))

# Home Page
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predictdata", methods = ["GET","POST"])
def predict_datapoint():
    if request.method == "POST":
       Temprature = float(request.form.get("Temperature"))
       RH = float(request.form.get("RH"))
       Ws = float(request.form.get("Ws"))
       Rain = float(request.form.get("Rain"))
       FFMC = float(request.form.get("FFMC"))
       DMC = float(request.form.get("DMC"))
       ISI = float(request.form.get("ISI"))
       Classes = float(request.form.get("Classes"))
       Region = float(request.form.get("Region"))

       # Create a numpy array with the input values
       input_data = np.array([[Temprature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]])

       # Standardize the input data
       scaled_data = standard_scalar.transform(input_data)

       # Make prediction using the loaded model
       prediction = ridge_model.predict(scaled_data)

       # Return the prediction result
       return render_template("home.html", result=prediction[0])
    else:
        return render_template("home.html")

if __name__ == "__main__":
    app.run(host= "0.0.0.0")