from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the ML model
model = joblib.load('crop app')

# Home route to render the form
@app.route('/')
def home():
    return render_template('index.html')  # Ensure index.html exists in the 'templates' folder

# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the form
        data = request.json
        arr = [
            float(data['nitrogen']),
            float(data['phosphorus']),
            float(data['potassium']),
            float(data['temperature']),
            float(data['humidity']),
            float(data['ph']),
            float(data['rainfall'])
        ]
        # Predict using the ML model
        prediction = model.predict([arr])
        return jsonify({'crop': prediction[0]})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
