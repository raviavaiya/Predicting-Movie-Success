from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd

# Initialize the Flask application
app = Flask(__name__)

# Load the pre-trained model
model = joblib.load('D:\\Predicting-Movie-Success\\Model\\random_forest_model.joblib')

# Define the feature names (ensure these match the order used during training)
feature_names = [
    'director_name', 'duration', 'gross', 'genres', 'actor_1_name',
    'num_voted_users', 'num_user_for_reviews', 'budget', 'title_year',
    'movie_facebook_likes'
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the request
    data = request.form.to_dict()

    # Ensure the input data has the correct number of features
    if len(data) != len(feature_names):
        return jsonify({'error': 'Input data must have 10 features'}), 400

    # Convert the input data to a DataFrame
    input_data = pd.DataFrame([data], columns=feature_names)

    # Make the prediction
    prediction = model.predict(input_data)

    # Convert the prediction result to "HIT", "AVG", or "FLOP"
    prediction_result = "HIT" if prediction[0] == 1 else "AVG" if prediction[0] == 2 else "FLOP"

    # Return the prediction as a JSON response
    return jsonify({'prediction': prediction_result})

if __name__ == '__main__':
    app.run(debug=True)
