from flask import Flask, request, jsonify
from flask_cors import CORS
from flask import Flask, request, jsonify
import joblib
import numpy as np

# Load model and scaler
model = joblib.load('random_forest_model (1).joblib')
scaler = joblib.load('scaler.joblib')

app = Flask(__name__)
CORS(app) 

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # Get JSON data from the request
    input_data = np.array(list(data.values())).reshape(1, -1)  # Reshape data
    scaled_data = scaler.transform(input_data)  # Scale the data
    prediction = model.predict(scaled_data)  # Make prediction
    probability = model.predict_proba(scaled_data)

    # Send prediction and probability as a JSON response
    return jsonify({'prediction': int(prediction[0]), 'probability': probability[0].tolist()})

if __name__ == '__main__':
    app.run(debug=True)
