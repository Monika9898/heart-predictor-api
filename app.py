from flask import Flask, request, jsonify
import joblib

# Load your model
model = joblib.load('heart_model.pkl')

# Start Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return "✅ Heart Prediction API is Running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    features = [
        data['age'], data['sex'], data['cp'], data['trestbps'], data['chol'],
        data['fbs'], data['restecg'], data['thalach'], data['exang'],
        data['oldpeak'], data['slope'], data['ca'], data['thal']
    ]

    prediction = model.predict([features])[0]
    result = 'Normal' if prediction == 0 else 'Abnormal'

    return jsonify({'prediction': result})

if __name__ == '__main__':
 app.run(host='0.0.0.0', port=5000, debug=True)
