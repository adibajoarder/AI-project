from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np
import pickle

app = Flask(__name__)

# Load model and type map
model = load_model("fraud_model.h5")
with open("type_map.pkl", "rb") as f:
    type_map = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        ttype = request.form['ttype']
        amount = float(request.form['amount'])
        oldbalanceOrg = float(request.form['oldbalanceOrg'])
        newbalanceOrig = float(request.form['newbalanceOrig'])
        oldbalanceDest = float(request.form['oldbalanceDest'])
        newbalanceDest = float(request.form['newbalanceDest'])

        ttype_encoded = type_map.get(ttype, 0)
        sample = np.array([[ttype_encoded, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest]])
        prob = model.predict(sample)[0][0]
        pred_text = "Fraud 🚨" if prob > 0.5 else "Not Fraud ✅"

        return render_template('result.html', prediction_text=pred_text, probability=prob)
    except Exception as e:
        return render_template('result.html', prediction_text=f"Error: {str(e)}", probability=0)

if __name__ == "__main__":
    app.run(debug=True)
